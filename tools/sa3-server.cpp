// sa3-server: a small HTTP server over the shared sa3 Pipeline — a PROOF OF CONCEPT that mirrors
// gary4local's async job model, so a gary4juce-style client can drive it as a drop-in (SA3 on :8006).
//   POST /generate {prompt, duration, steps, seed, negative_prompt, cfg_*, dist_shift,
//                   duration_padding_sec, loras[], init_path, keep_models}
//                 -> {success, session_id, seed}   (generation runs in the background)
//   POST /generate/loop {prompt, duration, bpm?, bars?, ...}
//                 -> {success, session_id, seed, bpm, bars, loop_duration, gen_duration}
//   GET  /poll_status/<session_id>
//                 -> {success, generation_in_progress, progress, step, total_steps, status,
//                     queue_status, audio_data (base64 wav, on "completed"), meta:{seed}}
//   POST /unload   -> free the model (full VRAM release; orchestrator owns the unload policy)
//   GET  /health   -> {status, model, encoding, loaded}
//   GET  /config   -> server defaults (the web frontend fetches this on startup)
// The Pipeline carries the reusable primitives (incl. GenParams::on_progress); a synchronous or SSE
// transport is left to real apps — this server only demonstrates the poll_status pattern.
#include "sa3_pipeline.h"
#include "sa3_config.h"
#include "embedded_web.h"
#include "wav.h"

#include "httplib.h"
#include "yyjson.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

std::mutex g_mtx;                         // serialize: one generation (one GPU graph) at a time
std::unique_ptr<sa3::Pipeline> g_pipe;    // loaded lazily on first generate; freed on /unload
std::atomic<bool> g_loaded{false};        // lock-free view for /health (won't block during a gen)

// Single source of truth for all config defaults (set from CLI flags at startup).
Sa3Config cfg;

// --- async job registry (mirrors gary4local /poll_status) ---
struct Job {
    std::string status = "queued";        // queued | generating | encoding | completed | failed | cancelled
    int      progress = 0;                // 0..100
    int      step = 0, total_steps = 0;
    std::string audio_b64;                // base64 wav, filled on completion
    std::string loudness_json;
    std::string error;
    uint64_t seed = 0;
    bool     cancelled = false;           // set by POST /cancel, polled by pipeline
    double   created = 0.0;
    double   finished = 0.0;
};
std::mutex jobs_mtx;
std::unordered_map<std::string, Job> jobs;

std::string json_escape(const std::string& s);
std::string json_err(const std::string& msg) { return "{\"error\":\"" + json_escape(msg) + "\"}"; }

constexpr int k_sample_rate = 44100;
constexpr int k_samples_per_latent_frame = 4096;

bool duration_to_target_samples(double duration, int& target_n_samp) {
    const double samples_d = std::round(duration * (double)k_sample_rate);
    if (samples_d < 1.0 || samples_d > (double)std::numeric_limits<int>::max()) return false;
    target_n_samp = (int)samples_d;
    return true;
}

int frames_for_target_samples(int target_n_samp) {
    int frames = (int)std::max<int64_t>(1, ((int64_t)target_n_samp + k_samples_per_latent_frame - 1) / k_samples_per_latent_frame);
    if (frames & 1) frames++;  // SAME-S needs even latent frame counts; harmless for SAME-L.
    return frames;
}

// minimal JSON string escaping (quotes/backslashes/control) for error text we splice into bodies
std::string json_escape(const std::string& s) {
    std::string o; o.reserve(s.size() + 8);
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (c == '\n') o += "\\n";
        else if (c == '\r') o += "\\r";
        else if (c == '\t') o += "\\t";
        else if ((unsigned char)c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
        else o += c;
    }
    return o;
}

std::string json_num(double v) {
    if (!std::isfinite(v)) return "null";
    char b[64];
    snprintf(b, sizeof b, "%.9g", v);
    return b;
}

std::string json_opt_num(bool enabled, double v) {
    return enabled ? json_num(v) : "null";
}

std::string loudness_params_json(const sa3::LoudnessParams& p) {
    std::string body = "{";
    body += "\"latent_rescale\":" + json_num(p.latent_rescale);
    body += ",\"latent_shift\":" + json_num(p.latent_shift);
    body += ",\"latent_target_std\":" + json_opt_num(p.latent_target_std_enabled, p.latent_target_std);
    body += ",\"latent_adapt_min\":" + json_num(p.latent_adapt_min);
    body += ",\"latent_adapt_max\":" + json_num(p.latent_adapt_max);
    body += ",\"peak_normalize_db\":" + json_opt_num(p.peak_normalize_enabled, p.peak_normalize_db);
    body += ",\"limiter_ceiling_db\":" + json_opt_num(p.limiter_enabled, p.limiter_ceiling_db);
    body += ",\"limiter_knee\":" + json_num(p.limiter_knee);
    body += "}";
    return body;
}

std::string loudness_meta_json(const sa3::LoudnessMeta& meta) {
    const sa3::LoudnessParams& p = meta.params;
    std::string body = "{";
    body += "\"latent_rescale\":" + json_num(p.latent_rescale);
    body += ",\"latent_shift\":" + json_num(p.latent_shift);
    body += ",\"latent_target_std\":" + json_opt_num(p.latent_target_std_enabled, p.latent_target_std);
    body += ",\"latent_adapt_min\":" + json_num(p.latent_adapt_min);
    body += ",\"latent_adapt_max\":" + json_num(p.latent_adapt_max);
    body += ",\"latent_factor\":" + json_num(meta.latent_factor);
    body += ",\"latent_std\":" + json_opt_num(meta.latent_std_set, meta.latent_std);
    body += ",\"peak_normalize_db\":" + json_opt_num(p.peak_normalize_enabled, p.peak_normalize_db);
    body += ",\"peak_normalize_gain\":" + json_opt_num(meta.peak_normalize_gain_set, meta.peak_normalize_gain);
    body += ",\"limiter_ceiling_db\":" + json_opt_num(p.limiter_enabled, p.limiter_ceiling_db);
    body += ",\"limiter_knee\":" + json_num(p.limiter_knee);
    body += ",\"limiter_limited_fraction\":" + json_opt_num(meta.limiter_limited_fraction_set, meta.limiter_limited_fraction);
    body += ",\"decoded_peak\":" + json_num(meta.decoded_peak);
    body += ",\"final_peak\":" + json_num(meta.final_peak);
    body += "}";
    return body;
}

bool starts_with(const std::string& s, const std::string& prefix) {
    return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string lower_ascii(std::string s) {
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

std::vector<std::string> split_dash(const std::string& s) {
    std::vector<std::string> parts;
    size_t start = 0;
    while (start <= s.size()) {
        size_t sep = s.find('-', start);
        parts.push_back(s.substr(start, sep == std::string::npos ? std::string::npos : sep - start));
        if (sep == std::string::npos) break;
        start = sep + 1;
    }
    return parts;
}

std::string join_dash(const std::vector<std::string>& parts) {
    std::string out;
    for (size_t i = 0; i < parts.size(); i++) {
        if (i) out += "-";
        out += parts[i];
    }
    return out;
}

bool looks_like_version_token(const std::string& token) {
    return token.size() >= 2 && (token[0] == 'v' || token[0] == 'V') && std::isdigit((unsigned char)token[1]);
}

std::string strip_lora_suffix_tokens(const std::string& raw) {
    std::vector<std::string> parts = split_dash(raw);
    static const std::set<std::string> trailing = {
        "lora", "f32", "f16", "q8_0", "q6_k", "q5_k_m", "q4_k_m"
    };
    while (!parts.empty()) {
        const std::string t = lower_ascii(parts.back());
        if (trailing.count(t) || looks_like_version_token(parts.back())) parts.pop_back();
        else break;
    }
    return join_dash(parts);
}

std::string infer_lora_name_from_filename(const std::filesystem::path& path) {
    const std::string stem = path.stem().string();
    if (starts_with(stem, "lora-")) return strip_lora_suffix_tokens(stem.substr(5));
    if (ends_with(lower_ascii(stem), "-lora")) return strip_lora_suffix_tokens(stem);
    return "";
}

std::string resolve_lora_path(const std::string& adapters_dir, const std::string& name_or_path) {
    namespace fs = std::filesystem;
    if (fs::exists(name_or_path) && lower_ascii(fs::path(name_or_path).extension().string()) == ".gguf") return name_or_path;

    const fs::path direct = fs::path(adapters_dir) / name_or_path;
    if (fs::exists(direct) && lower_ascii(direct.extension().string()) == ".gguf") return direct.string();

    const fs::path direct_gguf = fs::path(adapters_dir) / (name_or_path + ".gguf");
    if (fs::exists(direct_gguf)) return direct_gguf.string();

    std::string p = sa3::resolve_one(adapters_dir, "lora-" + name_or_path + "-", ".gguf");
    if (!p.empty()) return p;
    return sa3::resolve_one(adapters_dir, name_or_path + "-", ".gguf");
}

struct LoraEntry {
    std::string name;
    std::string path;
};

std::vector<LoraEntry> scan_loras(const std::string& adapters_dir) {
    namespace fs = std::filesystem;
    std::vector<LoraEntry> out;
    std::set<std::string> seen;
    std::error_code ec;
    if (!fs::is_directory(adapters_dir, ec)) return out;

    for (const auto& e : fs::directory_iterator(adapters_dir, ec)) {
        if (ec) break;
        if (!e.is_regular_file(ec)) continue;
        if (lower_ascii(e.path().extension().string()) != ".gguf") continue;
        std::string name = infer_lora_name_from_filename(e.path());
        if (name.empty()) continue;
        const std::string key = lower_ascii(name);
        if (!seen.insert(key).second) continue;
        out.push_back({name, e.path().string()});
    }
    std::sort(out.begin(), out.end(), [](const LoraEntry& a, const LoraEntry& b) {
        return lower_ascii(a.name) < lower_ascii(b.name);
    });
    return out;
}

std::string json_string_array(const std::vector<std::string>& values) {
    std::string body = "[";
    for (size_t i = 0; i < values.size(); i++) {
        if (i) body += ",";
        body += "\"" + json_escape(values[i]) + "\"";
    }
    body += "]";
    return body;
}

std::string loras_json(const std::vector<LoraEntry>& loras) {
    std::string body = "[";
    for (size_t i = 0; i < loras.size(); i++) {
        if (i) body += ",";
        body += "{\"index\":" + std::to_string(i)
             + ",\"name\":\"" + json_escape(loras[i].name)
             + "\",\"path\":\"" + json_escape(loras[i].path) + "\"}";
    }
    body += "]";
    return body;
}

bool read_text_file(const std::filesystem::path& path, std::string& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

using DiceMap = std::map<std::string, std::vector<std::string>>;

void add_unique_prompt(std::vector<std::string>& prompts, std::set<std::string>& seen, const std::string& prompt) {
    if (prompt.empty()) return;
    const std::string key = lower_ascii(prompt);
    if (!seen.insert(key).second) return;
    prompts.push_back(prompt);
}

bool load_dice_file(const std::filesystem::path& path, DiceMap& dice, int* version = nullptr) {
    std::string text;
    if (!read_text_file(path, text)) return false;
    yyjson_doc* doc = yyjson_read(text.c_str(), text.size(), 0);
    if (!doc) return false;
    yyjson_val* root = yyjson_doc_get_root(doc);
    if (version) {
        yyjson_val* v = yyjson_obj_get(root, "version");
        if (v && yyjson_is_int(v)) *version = (int)yyjson_get_int(v);
    }
    yyjson_val* dice_obj = yyjson_obj_get(root, "dice");
    if (!dice_obj || !yyjson_is_obj(dice_obj)) { yyjson_doc_free(doc); return false; }

    yyjson_obj_iter iter;
    yyjson_obj_iter_init(dice_obj, &iter);
    yyjson_val* key;
    while ((key = yyjson_obj_iter_next(&iter))) {
        yyjson_val* value = yyjson_obj_iter_get_val(key);
        if (!yyjson_is_str(key) || !yyjson_is_arr(value)) continue;
        const std::string bucket = yyjson_get_str(key);
        std::set<std::string> seen;
        auto& prompts = dice[bucket];
        for (const auto& existing : prompts) seen.insert(lower_ascii(existing));

        yyjson_val* item;
        yyjson_arr_iter ai;
        yyjson_arr_iter_init(value, &ai);
        while ((item = yyjson_arr_iter_next(&ai))) {
            if (yyjson_is_str(item)) add_unique_prompt(prompts, seen, yyjson_get_str(item));
        }
    }
    yyjson_doc_free(doc);
    return true;
}

std::vector<std::string> prompt_pool_names(const std::string& prompts_dir) {
    namespace fs = std::filesystem;
    std::vector<std::string> out;
    std::error_code ec;
    if (!fs::is_directory(prompts_dir, ec)) return out;
    for (const auto& e : fs::directory_iterator(prompts_dir, ec)) {
        if (ec) break;
        if (!e.is_regular_file(ec)) continue;
        if (lower_ascii(e.path().extension().string()) != ".json") continue;
        std::string name = lower_ascii(e.path().stem().string());
        if (name == "defaults") continue;
        out.push_back(name);
    }
    std::sort(out.begin(), out.end());
    return out;
}

std::vector<std::string> selected_lora_prompt_names(const httplib::Request& req) {
    std::vector<std::string> out;
    std::set<std::string> seen;
    const size_t count = req.get_param_value_count("lora");
    for (size_t i = 0; i < count; i++) {
        std::string value = req.get_param_value("lora", i);
        size_t start = 0;
        while (start <= value.size()) {
            size_t sep = value.find(',', start);
            std::string name = lower_ascii(value.substr(start, sep == std::string::npos ? std::string::npos : sep - start));
            name.erase(std::remove_if(name.begin(), name.end(), [](unsigned char c) { return std::isspace(c); }), name.end());
            if (!name.empty() && seen.insert(name).second) out.push_back(name);
            if (sep == std::string::npos) break;
            start = sep + 1;
        }
    }
    return out;
}

std::string dice_json(const DiceMap& dice) {
    std::string body = "{";
    bool first_bucket = true;
    for (const auto& kv : dice) {
        if (!first_bucket) body += ",";
        first_bucket = false;
        body += "\"" + json_escape(kv.first) + "\":" + json_string_array(kv.second);
    }
    body += "}";
    return body;
}

std::string b64_encode(const std::string& in) {
    static const char* T = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out; out.reserve(((in.size() + 2) / 3) * 4);
    size_t i = 0;
    for (; i + 3 <= in.size(); i += 3) {
        unsigned n = (unsigned char)in[i] << 16 | (unsigned char)in[i+1] << 8 | (unsigned char)in[i+2];
        out += T[(n>>18)&63]; out += T[(n>>12)&63]; out += T[(n>>6)&63]; out += T[n&63];
    }
    if (i < in.size()) {
        unsigned n = (unsigned char)in[i] << 16;
        if (i + 1 < in.size()) n |= (unsigned char)in[i+1] << 8;
        out += T[(n>>18)&63]; out += T[(n>>12)&63];
        out += (i + 1 < in.size()) ? T[(n>>6)&63] : '=';
        out += '=';
    }
    return out;
}

std::string new_session_id() {
    static const char* H = "0123456789abcdef";
    std::random_device rd;
    uint64_t r = ((uint64_t)rd() << 32) ^ (uint64_t)rd();
    std::string s(12, '0');
    for (int i = 0; i < 12; i++) { s[i] = H[r & 0xF]; r >>= 4; }
    return s;
}

// Drop finished jobs so the registry does not keep large base64 WAV payloads indefinitely
// (call under jobs_mtx). The preferred client path is /poll_status/<id>?consume=1, which
// removes completed jobs immediately after the successful audio fetch.
void jobs_prune() {
    constexpr double kFinishedJobTtlS = 600.0;  // 10 min, matches docs/SERVER.md
    const double now = sa3::wall_time_s();
    for (auto it = jobs.begin(); it != jobs.end(); ) {
        const bool done = it->second.status == "completed" || it->second.status == "failed";
        const double since = now - (it->second.finished > 0.0 ? it->second.finished : it->second.created);
        if (done && since > kFinishedJobTtlS) it = jobs.erase(it);
        else ++it;
    }
}

// (re)load the pipeline under the caller's g_mtx. Returns false + message on failure.
bool ensure_loaded(std::string& err) {
    if (g_pipe && g_pipe->loaded()) { g_loaded = true; return true; }
    sa3::ModelPaths mp;
    if (!sa3::ModelPaths::resolve(cfg.models_dir, cfg.model_variant, cfg.encoding, mp, err)) return false;
    try {
        g_pipe = std::make_unique<sa3::Pipeline>();
        g_pipe->load(mp, cfg.cpu_threads, cfg.device.empty() ? nullptr : cfg.device.c_str(), cfg.gpu_selector.empty() ? nullptr : cfg.gpu_selector.c_str());
    } catch (const std::exception& e) { g_pipe.reset(); g_loaded = false; err = e.what(); return false; }
    g_loaded = true;
    return true;
}

bool valid_loop_bars(int bars) {
    return bars == 4 || bars == 8 || bars == 16 || bars == 32;
}

float extract_bpm_from_prompt(const std::string& prompt) {
    const std::string lower = lower_ascii(prompt);
    size_t pos = lower.find("bpm");
    while (pos != std::string::npos) {
        size_t end = pos;
        while (end > 0 && std::isspace((unsigned char)lower[end - 1])) end--;
        size_t start = end;
        while (start > 0) {
            const char c = lower[start - 1];
            if ((c >= '0' && c <= '9') || c == '.') start--;
            else break;
        }
        if (start < end) {
            float bpm = 0.0f;
            if (sa3::parse_float_text(prompt.substr(start, end - start).c_str(), bpm)) return bpm;
        }
        pos = lower.find("bpm", pos + 3);
    }
    return 0.0f;
}

bool parse_generate_request(yyjson_val* root, const std::string& adir,
                            sa3::GenParams& params, uint64_t& seed_resolved,
                            std::string& perr, const Sa3Config& cfg) {
    auto S = [&](const char* k, const char* d) { yyjson_val* v = yyjson_obj_get(root, k); return std::string(v && yyjson_is_str(v) ? yyjson_get_str(v) : d); };
    auto I = [&](const char* k, int d) {
        yyjson_val* v = yyjson_obj_get(root, k);
        if (v && yyjson_is_int(v)) return (int)yyjson_get_int(v);
        if (v && yyjson_is_num(v)) return (int)yyjson_get_num(v);
        return d;
    };
    auto D = [&](const char* k, double d) { yyjson_val* v = yyjson_obj_get(root, k); return v && yyjson_is_num(v) ? yyjson_get_num(v) : d; };
    auto B = [&](const char* k, bool d) { yyjson_val* v = yyjson_obj_get(root, k); return v && yyjson_is_bool(v) ? yyjson_get_bool(v) : d; };

    params.prompt           = S("prompt", "");
    const std::string init_path = S("init_path", "");
    if (yyjson_obj_get(root, "seconds")) {
        perr = "use duration, not seconds";
        return false;
    }
    if (yyjson_obj_get(root, "frames")) {
        perr = "HTTP API uses duration seconds; frames is CLI/internal";
        return false;
    }
    const double max_duration = D("max_duration", cfg.max_duration);
    const double duration = D("duration", cfg.duration);
    if (!std::isfinite(duration) || duration <= 0.0 || duration > max_duration) {
        perr = "duration must be in (0, " + json_num(max_duration) + "] seconds";
        return false;
    }
    int duration_target_samples = 0;
    if (!duration_to_target_samples(duration, duration_target_samples)) {
        perr = "duration is out of range";
        return false;
    }
    params.frames = frames_for_target_samples(duration_target_samples);
    params.steps            = I("steps", cfg.steps);
    seed_resolved           = sa3::pick_seed(I("seed", 0));   // seed -1 => random
    params.seed             = seed_resolved;
    params.keep_models      = B("keep_models", false);        // FRUGAL default
    params.init_noise_level = (float)D("init_noise_level", cfg.init_noise_level);
    params.inpaint_start    = (float)D("inpaint_start", -1.0);
    params.inpaint_end      = (float)D("inpaint_end", -1.0);
    params.duration_padding_sec = (float)D("duration_padding_sec", 6.0);
    params.target_n_samp    = I("target_samples", init_path.empty() ? duration_target_samples : 0);
    if (params.target_n_samp < 0) params.target_n_samp = 0;
    else if (params.target_n_samp > params.frames * k_samples_per_latent_frame) params.target_n_samp = params.frames * k_samples_per_latent_frame;
    params.encode_chunk_size = I("encode_chunk_size", 0);
    params.encode_overlap    = I("encode_overlap", 32);
    params.decode_chunk_size = I("decode_chunk_size", 0);
    params.decode_overlap    = I("decode_overlap", 32);
    params.loudness          = cfg.loudness;

    auto parse_json_float = [&](yyjson_val* v, float& out, const char* key) {
        if (!v) return true;
        if (yyjson_is_num(v)) {
            out = (float)yyjson_get_num(v);
            if (std::isfinite(out)) return true;
        }
        if (yyjson_is_str(v)) {
            if (sa3::parse_float_text(yyjson_get_str(v), out)) return true;
        }
        if (perr.empty()) perr = std::string(key) + " must be a finite number";
        return false;
    };
    auto request_float = [&](const char* key, float& dst) {
        if (!perr.empty()) return;
        yyjson_val* v = yyjson_obj_get(root, key);
        if (!v) return;
        parse_json_float(v, dst, key);
    };
    auto request_optional_float = [&](const char* key, bool& enabled, float& dst, bool positive_disables = false) {
        if (!perr.empty()) return;
        yyjson_val* v = yyjson_obj_get(root, key);
        if (!v) return;
        if (yyjson_is_null(v) || (yyjson_is_bool(v) && !yyjson_get_bool(v))) {
            enabled = false;
            return;
        }
        if (yyjson_is_str(v) && sa3::text_disables_optional_float(yyjson_get_str(v))) {
            enabled = false;
            return;
        }
        float parsed = dst;
        if (!parse_json_float(v, parsed, key)) return;
        if (positive_disables && parsed > 0.0f) {
            enabled = false;
            return;
        }
        enabled = true;
        dst = parsed;
    };
    request_float("latent_rescale", params.loudness.latent_rescale);
    request_float("latent_shift", params.loudness.latent_shift);
    request_optional_float("latent_target_std", params.loudness.latent_target_std_enabled, params.loudness.latent_target_std);
    request_float("latent_adapt_min", params.loudness.latent_adapt_min);
    request_float("latent_adapt_max", params.loudness.latent_adapt_max);
    request_optional_float("peak_normalize_db", params.loudness.peak_normalize_enabled, params.loudness.peak_normalize_db);
    request_optional_float("limiter_ceiling_db", params.loudness.limiter_enabled, params.loudness.limiter_ceiling_db, true);
    request_float("limiter_knee", params.loudness.limiter_knee);
    sa3::normalize_loudness_params(params.loudness);

    params.negative_prompt   = S("negative_prompt", "");
    params.cfg_scale         = (float)D("cfg_scale", cfg.cfg_scale);
    params.cfg_rescale       = (float)D("cfg_rescale", 0.0);
    params.apg_scale         = (float)D("apg_scale", 1.0);
    params.cfg_norm_threshold= (float)D("cfg_norm_threshold", 0.0);
    params.cfg_interval_min  = (float)D("cfg_interval_min", 0.0);
    params.cfg_interval_max  = (float)D("cfg_interval_max", 1.0);

    params.dist_shift       = S("dist_shift", "LogSNR");
    sa3::dist_shift_defaults(params.dist_shift, params.ds_p1, params.ds_p2, params.ds_p3, params.ds_p4);
    if (yyjson_val* dsp = yyjson_obj_get(root, "dist_shift_params"); dsp && yyjson_is_arr(dsp)) {
        float* slots[4] = { &params.ds_p1, &params.ds_p2, &params.ds_p3, &params.ds_p4 };
        yyjson_val* v; yyjson_arr_iter di; yyjson_arr_iter_init(dsp, &di);
        for (int k = 0; k < 4 && (v = yyjson_arr_iter_next(&di)); k++)
            if (yyjson_is_num(v)) *slots[k] = (float)yyjson_get_num(v);
    }

    yyjson_val* loras = yyjson_obj_get(root, "loras");
    if (loras && yyjson_is_arr(loras)) {
        yyjson_val* it; yyjson_arr_iter ai; yyjson_arr_iter_init(loras, &ai);
        while ((it = yyjson_arr_iter_next(&ai))) {
            yyjson_val* nv = yyjson_obj_get(it, "name");
            if (!nv) nv = yyjson_obj_get(it, "path");
            std::string name = nv && yyjson_is_str(nv) ? yyjson_get_str(nv) : "";
            yyjson_val* sv = yyjson_obj_get(it, "strength");
            float strength = sv && yyjson_is_num(sv) ? (float)yyjson_get_num(sv) : 1.0f;
            if (name.empty()) continue;
            std::string p = resolve_lora_path(adir, name);
            if (p.empty()) { perr = "unknown lora '" + name + "'"; break; }
            params.loras.push_back({p, strength});
        }
    }

    if (!perr.empty()) return false;
    if (!sa3::validate_loudness_params(params.loudness, perr)) return false;

    if (!init_path.empty()) {
        if (!std::filesystem::exists(init_path)) {
            perr = "init_path not found: " + init_path;
            return false;
        }
        int ns = 0, nc = 0, sr = 0;
        try {
            params.init_audio = sa3::read_wav_planar(init_path, ns, nc, sr);
        } catch (const std::exception& e) {
            perr = std::string("invalid init_path WAV: ") + e.what();
            return false;
        }
        params.init_n_samp = ns; params.init_n_ch = nc; params.init_sample_rate = sr;
    }

    return true;
}

std::string queue_generation(sa3::GenParams params, uint64_t seed_resolved) {
    const std::string sid = new_session_id();
    {
        std::lock_guard<std::mutex> lk(jobs_mtx);
        jobs_prune();
        Job j; j.total_steps = params.steps; j.seed = seed_resolved; j.created = sa3::wall_time_s();
        jobs[sid] = std::move(j);
    }
    const std::string peak_norm_log = params.loudness.peak_normalize_enabled
        ? json_num(params.loudness.peak_normalize_db) : "off";
    const std::string limiter_log = params.loudness.limiter_enabled
        ? json_num(params.loudness.limiter_ceiling_db) : "off";
    fprintf(stderr,
            "[sa3-server] queued %s frames=%d (~%.2fs) target_samples=%d steps=%d keep_models=%s init_samples=%d init_ch=%d inpaint=%.2f..%.2f init_noise=%.2f loras=%zu ae_chunks=enc%d/%d dec%d/%d peak_norm=%s limiter=%s\n",
            sid.c_str(), params.frames, (double)params.frames * 4096.0 / 44100.0, params.target_n_samp,
            params.steps, params.keep_models ? "true" : "false", params.init_n_samp, params.init_n_ch,
            params.inpaint_start, params.inpaint_end, params.init_noise_level, params.loras.size(),
            params.encode_chunk_size, params.encode_overlap, params.decode_chunk_size, params.decode_overlap,
            peak_norm_log.c_str(), limiter_log.c_str());
    fflush(stderr);
    std::thread([sid, seed_resolved, params = std::move(params)]() mutable {
        params.on_progress = [sid](const sa3::Progress& p) {
            fprintf(stderr, "[sa3-server] job %s %s %d/%d %.0f%%\n",
                    sid.c_str(), p.stage, p.step, p.total, p.fraction * 100.0f);
            fflush(stderr);
            std::lock_guard<std::mutex> lk(jobs_mtx);
            auto it = jobs.find(sid); if (it == jobs.end()) return;
            it->second.progress    = (int)(p.fraction * 100.0f);
            it->second.step        = p.step;
            it->second.total_steps = p.total;
            if      (!strcmp(p.stage, "sampling")) it->second.status = "generating";
            else if (!strcmp(p.stage, "encoding")) it->second.status = "encoding";
            else if (!strcmp(p.stage, "decoding")) it->second.status = "decoding";
            else if (!strcmp(p.stage, "done"))     it->second.status = "finalizing";
        };
        params.should_cancel = [sid]() {
            std::lock_guard<std::mutex> jl(jobs_mtx);
            auto it = jobs.find(sid);
            return it != jobs.end() && it->second.cancelled;
        };
        std::lock_guard<std::mutex> lk(g_mtx);
        std::string err;
        fprintf(stderr, "[sa3-server] job %s starting\n", sid.c_str());
        fflush(stderr);
        if (!ensure_loaded(err)) {
            fprintf(stderr, "[sa3-server] job %s failed to load model: %s\n", sid.c_str(), err.c_str());
            fflush(stderr);
            std::lock_guard<std::mutex> jl(jobs_mtx);
            if (auto it = jobs.find(sid); it != jobs.end()) { it->second.status = "failed"; it->second.error = err; }
            return;
        }
        fprintf(stderr, "[sa3-server] job %s model ready\n", sid.c_str());
        fflush(stderr);
        try {
            sa3::GenResult r = g_pipe->generate(params);
            std::string b64 = b64_encode(sa3::wav_planar_bytes(r.samples.data(), r.n_samp, r.n_ch, r.sample_rate));
            std::lock_guard<std::mutex> jl(jobs_mtx);
            if (auto it = jobs.find(sid); it != jobs.end()) {
                it->second.audio_b64 = std::move(b64);
                it->second.loudness_json = loudness_meta_json(r.loudness);
                it->second.status = "completed"; it->second.progress = 100;
                it->second.finished = sa3::wall_time_s();
            }
            fprintf(stderr, "[sa3-server] job %s completed %.2fs seed=%llu peak %.3f -> %.3f limited %.4f%%\n",
                    sid.c_str(), (double)r.n_samp / (double)r.sample_rate, (unsigned long long)seed_resolved,
                    r.loudness.decoded_peak, r.loudness.final_peak,
                    r.loudness.limiter_limited_fraction_set ? 100.0 * r.loudness.limiter_limited_fraction : 0.0);
            fflush(stderr);
        } catch (const std::exception& e) {
            fprintf(stderr, "[sa3-server] job %s failed: %s\n", sid.c_str(), e.what());
            fflush(stderr);
            std::lock_guard<std::mutex> jl(jobs_mtx);
            if (auto it = jobs.find(sid); it != jobs.end()) {
                it->second.status = it->second.cancelled ? "cancelled" : "failed";
                it->second.error = e.what();
                it->second.finished = sa3::wall_time_s();
            }
        }
    }).detach();
    return sid;
}

std::string config_json(const Sa3Config& cfg) {
    std::string body = "{";
    body += "\"model_variant\":\"" + json_escape(cfg.model_variant) + "\"";
    body += ",\"encoding\":\"" + json_escape(cfg.encoding) + "\"";
    body += ",\"models_dir\":\"" + json_escape(cfg.models_dir) + "\"";
    body += ",\"adapters_dir\":\"" + json_escape(cfg.adapters_dir) + "\"";
    body += ",\"prompts_dir\":\"" + json_escape(cfg.prompts_dir) + "\"";
    body += ",\"audio_in_dir\":\"" + json_escape(cfg.audio_in_dir) + "\"";
    body += ",\"duration\":" + json_num(cfg.duration);
    body += ",\"max_duration\":" + json_num(cfg.max_duration);
    body += ",\"steps\":" + std::to_string(cfg.steps);
    body += ",\"default_loop_bars\":" + std::to_string(cfg.default_loop_bars);
    body += ",\"loop_pad_seconds\":" + json_num(cfg.loop_pad_seconds);
    body += ",\"cfg_scale\":" + json_num(cfg.cfg_scale);
    body += ",\"cfg_rescale\":" + json_num(cfg.cfg_rescale);
    body += ",\"apg_scale\":" + json_num(cfg.apg_scale);
    body += ",\"cfg_norm_threshold\":" + json_num(cfg.cfg_norm_threshold);
    body += ",\"cfg_interval_min\":" + json_num(cfg.cfg_interval_min);
    body += ",\"cfg_interval_max\":" + json_num(cfg.cfg_interval_max);
    body += ",\"init_noise_level\":" + json_num(cfg.init_noise_level);
    body += ",\"inpaint_start\":" + json_num(cfg.inpaint_start);
    body += ",\"inpaint_end\":" + json_num(cfg.inpaint_end);
    body += ",\"seed\":" + std::to_string(cfg.seed);
    body += ",\"bpm\":" + json_num(cfg.bpm);
    body += ",\"flash_attn\":" + std::to_string(cfg.flash_attn);
    body += ",\"same_flash_attn_mode\":" + std::to_string(cfg.same_flash_attn_mode);
    body += ",\"encode_chunk_size\":" + std::to_string(cfg.encode_chunk_size);
    body += ",\"encode_overlap\":" + std::to_string(cfg.encode_overlap);
    body += ",\"decode_chunk_size\":" + std::to_string(cfg.decode_chunk_size);
    body += ",\"decode_overlap\":" + std::to_string(cfg.decode_overlap);
    body += ",\"cpu_threads\":" + std::to_string(cfg.cpu_threads);
    body += ",\"device\":\"" + json_escape(cfg.device) + "\"";
    body += ",\"gpu_selector\":\"" + json_escape(cfg.gpu_selector) + "\"";
    body += ",\"profile\":" + std::to_string(cfg.profile);
    body += ",\"loudness\":" + loudness_params_json(cfg.loudness);
    body += ",\"dist_shift_defaults\":{";
    body += "\"LogSNR\":[2000,-6.2,0,2]";
    body += ",\"Flux\":[256,4096,6.93,6.93]";
    body += ",\"Full\":[0.5,1.15,256,4096]";
    body += ",\"None\":[0,0,0,0]";
    body += "}";
    body += "}";
    return body;
}

} // namespace

int main(int argc, char** argv) {
    std::string host = cfg.host;
    int port = cfg.port;


    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto val = [&]{ return std::string(i + 1 < argc ? argv[++i] : ""); };
        if (a == "--host")              { auto v = val(); if (!v.empty()) host = v; }
        else if (a == "--port")         { auto v = val(); if (!v.empty()) port = atoi(v.c_str()); }
        else if (a == "--model")        { auto v = val(); if (!v.empty()) cfg.model_variant = v; }
        else if (a == "--encoding")     { auto v = val(); if (!v.empty()) cfg.encoding = v; }
        else if (a == "--models-dir")   { auto v = val(); if (!v.empty()) cfg.models_dir = v; }
        else if (a == "--adapters-dir") { auto v = val(); if (!v.empty()) cfg.adapters_dir = v; }
        else if (a == "--prompts-dir")  { auto v = val(); if (!v.empty()) cfg.prompts_dir = v; }
        else if (a == "--audio-in-dir") { auto v = val(); if (!v.empty()) cfg.audio_in_dir = v; }
        else if (a == "--device")       { auto v = val(); if (!v.empty()) cfg.device = v; }
        else if (a == "--gpu")          { auto v = val(); if (!v.empty()) cfg.gpu_selector = v; }
        else if (a == "--threads")      { auto v = val(); if (!v.empty()) { cfg.cpu_threads = atoi(v.c_str()); if (cfg.cpu_threads <= 0) { fprintf(stderr, "--threads must be positive\n"); return 1; } } }
        else if (a == "--flash-attn")   { auto v = val(); if (!v.empty()) cfg.flash_attn = atoi(v.c_str()); }
        else if (a == "--same-flash-attn") { auto v = val(); if (!v.empty()) cfg.same_flash_attn_mode = atoi(v.c_str()); }
        else if (a == "--profile")      { auto v = val(); if (!v.empty()) cfg.profile = atoi(v.c_str()); }
        else if (a == "--dump-cond")    { auto v = val(); if (!v.empty()) cfg.dump_cond_dir = v; }
        else if (a == "--help" || a == "-h") {
            fprintf(stdout, "usage: sa3-server [--host ADDR] [--port N] [--models-dir DIR] [--model medium|small-music|small-sfx]\n"
                            "                     [--encoding f16|f32] [--device DEVICE] [--gpu SELECTOR] [--threads N]\n"
                            "                     [--flash-attn 0|1] [--same-flash-attn 0|1|2] [--profile 0|1] [--dump-cond DIR]\n");
            return 0;
        }
        else { fprintf(stderr, "error: unrecognized flag '%s'\n", a.c_str()); return 1; }
    }

    // Apply config to library globals
    sa3::nn::g_flash_attn_enabled = cfg.flash_attn != 0;
    sa3::nn::g_same_flash_attn_mode = cfg.same_flash_attn_mode;
    sa3::g_profile_enabled = cfg.profile != 0;
    sa3::g_dump_cond_dir = cfg.dump_cond_dir;

    const std::string adir = cfg.adapters_dir;
    const std::string pdir = cfg.prompts_dir;
    const std::string aidir = cfg.audio_in_dir;

    // create audio-in directory if it doesn't exist
    std::error_code ec;
    std::filesystem::create_directories(aidir, ec);
    if (ec) fprintf(stderr, "[sa3-server] WARNING: could not create audio-in dir %s: %s\n", aidir.c_str(), ec.message().c_str());

    httplib::Server svr;

    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        const bool loaded = g_loaded.load();
        std::string body = "{\"status\":\"ok\",\"model\":\"" + cfg.model_variant + "\",\"encoding\":\"" +
                           cfg.encoding + "\",\"loaded\":" + (loaded ? "true" : "false") +
                           ",\"loudness_defaults\":" + loudness_params_json(cfg.loudness) + "}";
        res.set_content(body, "application/json");
    });

    svr.Get("/config", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(config_json(cfg), "application/json");
    });

    svr.Get("/loras", [&adir](const httplib::Request&, httplib::Response& res) {
        const std::vector<LoraEntry> loras = scan_loras(adir);
        std::string body = "{\"success\":true,\"loras\":" + loras_json(loras)
                         + ",\"adapters_dir\":\"" + json_escape(adir)
                         + "\",\"model_loaded\":" + (g_loaded.load() ? "true" : "false") + "}";
        res.set_content(body, "application/json");
    });

    svr.Get("/prompts", [&pdir](const httplib::Request& req, httplib::Response& res) {
        namespace fs = std::filesystem;
        DiceMap dice;
        int version = 1;
        load_dice_file(fs::path(pdir) / "defaults.json", dice, &version);
        if (dice.empty()) {
            dice["generic"] = {};
            dice["instrumental"] = {};
            dice["drums"] = {};
        }

        const std::vector<std::string> selected = selected_lora_prompt_names(req);
        std::vector<std::string> missing;
        std::set<std::string> replaced;

        for (const std::string& name : selected) {
            DiceMap lora_dice;
            if (!load_dice_file(fs::path(pdir) / (name + ".json"), lora_dice)) {
                missing.push_back(name);
                continue;
            }

            for (const auto& kv : lora_dice) {
                if (!replaced.count(kv.first)) {
                    dice[kv.first].clear();
                    replaced.insert(kv.first);
                }
                std::set<std::string> seen;
                for (const auto& existing : dice[kv.first]) seen.insert(lower_ascii(existing));
                for (const auto& prompt : kv.second) add_unique_prompt(dice[kv.first], seen, prompt);
            }
        }

        std::string body = "{\"success\":true,\"loras\":" + json_string_array(selected)
                         + ",\"missing_loras\":" + json_string_array(missing)
                         + ",\"available_loras\":" + json_string_array(prompt_pool_names(pdir))
                         + ",\"prompts\":{\"version\":" + std::to_string(version)
                         + ",\"dice\":" + dice_json(dice)
                         + ",\"source\":{\"generic\":\""
                         + json_escape(fs::exists(fs::path(pdir) / "defaults.json") ? "defaults.json" : "empty")
                         + "\"}}}";
        res.set_content(body, "application/json");
    });

    svr.Post("/unload", [](const httplib::Request&, httplib::Response& res) {
        std::lock_guard<std::mutex> lk(g_mtx);
        g_pipe.reset();
        g_loaded = false;
        res.set_content("{\"status\":\"unloaded\"}", "application/json");
    });

    svr.Post("/generate", [&adir](const httplib::Request& req, httplib::Response& res) {
        yyjson_doc* doc = yyjson_read(req.body.c_str(), req.body.size(), 0);
        if (!doc) { res.status = 400; res.set_content(json_err("invalid json"), "application/json"); return; }
        yyjson_val* root = yyjson_doc_get_root(doc);

        sa3::GenParams params;
        uint64_t seed_resolved = 0;
        std::string perr;
        if (!parse_generate_request(root, adir, params, seed_resolved, perr, cfg)) {
            yyjson_doc_free(doc);
            res.status = 400; res.set_content(json_err(perr), "application/json"); return;
        }
        yyjson_doc_free(doc);

        const std::string sid = queue_generation(std::move(params), seed_resolved);
        res.set_content("{\"success\":true,\"session_id\":\"" + sid + "\",\"seed\":" + std::to_string(seed_resolved) + "}", "application/json");
    });

    svr.Post("/generate/loop", [&adir](const httplib::Request& req, httplib::Response& res) {
        yyjson_doc* doc = yyjson_read(req.body.c_str(), req.body.size(), 0);
        if (!doc) { res.status = 400; res.set_content(json_err("invalid json"), "application/json"); return; }
        yyjson_val* root = yyjson_doc_get_root(doc);
        auto I = [&](const char* k, int d) {
            yyjson_val* v = yyjson_obj_get(root, k);
            if (v && yyjson_is_int(v)) return (int)yyjson_get_int(v);
            if (v && yyjson_is_num(v)) return (int)yyjson_get_num(v);
            return d;
        };

        auto D = [&](const char* k, double d) {
            yyjson_val* v = yyjson_obj_get(root, k);
            return v && yyjson_is_num(v) ? yyjson_get_num(v) : d;
        };

        sa3::GenParams params;
        uint64_t seed_resolved = 0;
        std::string perr;
        if (!parse_generate_request(root, adir, params, seed_resolved, perr, cfg)) {
            yyjson_doc_free(doc);
            res.status = 400; res.set_content(json_err(perr), "application/json"); return;
        }

        float bpm = (float)cfg.bpm;
        if (yyjson_val* bv = yyjson_obj_get(root, "bpm")) {
            if (yyjson_is_num(bv)) bpm = (float)yyjson_get_num(bv);
            else if (yyjson_is_str(bv)) sa3::parse_float_text(yyjson_get_str(bv), bpm);
        }
        if (bpm <= 0.0f) bpm = extract_bpm_from_prompt(params.prompt);

        const double loop_pad = D("loop_pad_seconds", cfg.loop_pad_seconds);
        const int bars = I("bars", cfg.default_loop_bars);
        yyjson_doc_free(doc);

        if (bpm <= 0.0f) {
            res.status = 400; res.set_content(json_err("BPM required in prompt or bpm field"), "application/json"); return;
        }
        if (!valid_loop_bars(bars)) {
            res.status = 400; res.set_content(json_err("bars must be one of [4, 8, 16, 32]"), "application/json"); return;
        }

        const double seconds_per_bar = (60.0 / (double)bpm) * 4.0;
        const double loop_duration = seconds_per_bar * (double)bars;
        const double gen_duration = loop_duration + loop_pad;
        const double max_duration = D("max_duration", cfg.max_duration);
        if (gen_duration > max_duration) {
            res.status = 400;
            res.set_content(json_err(std::to_string(bars) + " bars at " + json_num(bpm) + " bpm exceeds max " + json_num(max_duration) + "s with pad"), "application/json");
            return;
        }

        int target_samples = 0;
        int gen_samples = 0;
        if (!duration_to_target_samples(loop_duration, target_samples) ||
            !duration_to_target_samples(gen_duration, gen_samples)) {
            res.status = 400;
            res.set_content(json_err("loop duration is out of range"), "application/json");
            return;
        }
        params.frames = frames_for_target_samples(gen_samples);
        params.target_n_samp = target_samples;
        params.duration_padding_sec = 0.0f;

        const std::string sid = queue_generation(std::move(params), seed_resolved);
        std::string body = "{\"success\":true,\"session_id\":\"" + sid + "\",\"seed\":" + std::to_string(seed_resolved);
        body += ",\"bpm\":" + json_num(bpm);
        body += ",\"bars\":" + std::to_string(bars);
        body += ",\"seconds_per_bar\":" + json_num(seconds_per_bar);
        body += ",\"loop_duration\":" + json_num(loop_duration);
        body += ",\"gen_duration\":" + json_num(gen_duration);
        body += ",\"target_samples\":" + std::to_string(target_samples) + "}";
        res.set_content(body, "application/json");
    });

    svr.Get(R"(/poll_status/(.+))", [](const httplib::Request& req, httplib::Response& res) {
        const std::string sid = req.matches[1];
        const bool consume = req.has_param("consume") && req.get_param_value("consume") != "0";

        std::string status;
        int progress = 0, step = 0, total_steps = 0;
        uint64_t seed = 0;
        std::string audio_b64;
        std::string loudness_json;
        std::string error;
        {
            std::lock_guard<std::mutex> lk(jobs_mtx);
            jobs_prune();
            auto it = jobs.find(sid);
            if (it == jobs.end()) {
                res.status = 404;
                res.set_content("{\"success\":false,\"error\":\"unknown session: " + json_escape(sid) + "\"}", "application/json");
                return;
            }

            Job& j = it->second;
            status = j.status;
            progress = j.progress;
            step = j.step;
            total_steps = j.total_steps;
            seed = j.seed;
            error = j.error;
            loudness_json = j.loudness_json;

            if (j.status == "completed") {
                if (consume) {
                    audio_b64.swap(j.audio_b64);
                    loudness_json.swap(j.loudness_json);
                    jobs.erase(it);
                } else {
                    audio_b64 = j.audio_b64;
                }
            }
        }

        const bool in_prog = status == "queued" || status == "generating" || status == "encoding"
                             || status == "decoding" || status == "finalizing";
        std::string qs = status == "queued"
            ? "{\"status\":\"queued\",\"position\":1,\"total_queued\":1,\"message\":\"queued locally\",\"estimated_seconds\":5}"
            : in_prog ? "{\"status\":\"ready\"}" : "{}";
        std::string body = "{";
        body += "\"success\":" + std::string(status == "failed" ? "false" : "true") + ",";
        body += "\"generation_in_progress\":" + std::string(in_prog ? "true" : "false") + ",";
        body += "\"transform_in_progress\":false,";
        body += "\"progress\":" + std::to_string(progress) + ",";
        body += "\"step\":" + std::to_string(step) + ",";
        body += "\"total_steps\":" + std::to_string(total_steps) + ",";
        body += "\"status\":\"" + status + "\",";
        body += "\"queue_status\":" + qs;
        if (status == "completed")
            body += ",\"audio_data\":\"" + audio_b64 + "\",\"meta\":{\"seed\":" + std::to_string(seed) +
                    ",\"loudness\":" + (loudness_json.empty() ? "{}" : loudness_json) + "}";
        if (status == "failed" || status == "cancelled")
            body += ",\"error\":\"" + json_escape(error) + "\"";
        body += "}";
        res.set_content(body, "application/json");
    });

    svr.Post(R"(/cancel/(.+))", [](const httplib::Request& req, httplib::Response& res) {
        const std::string sid = req.matches[1];
        std::lock_guard<std::mutex> lk(jobs_mtx);
        auto it = jobs.find(sid);
        if (it == jobs.end()) {
            res.status = 404;
            res.set_content("{\"success\":false,\"error\":\"unknown session\"}", "application/json");
            return;
        }
        it->second.cancelled = true;
        fprintf(stderr, "[sa3-server] cancel requested for job %s\n", sid.c_str());
        fflush(stderr);
        res.set_content("{\"success\":true}", "application/json");
    });

    svr.Get("/audio-in", [&aidir](const httplib::Request&, httplib::Response& res) {
        namespace fs = std::filesystem;
        std::string body = "{\"success\":true,\"files\":[";
        bool first = true;
        if (fs::is_directory(aidir)) {
            for (const auto& entry : fs::directory_iterator(aidir)) {
                if (!entry.is_regular_file()) continue;
                std::string ext = entry.path().extension().string();
                for (auto& c : ext) c = (char)std::tolower((unsigned char)c);
                if (ext != ".wav") continue;
                if (!first) body += ",";
                first = false;
                body += "\"" + json_escape(entry.path().filename().string()) + "\"";
            }
        }
        body += "]}";
        res.set_content(body, "application/json");
    });

    svr.Post("/upload", [&aidir](const httplib::Request& req, httplib::Response& res) {
        if (!req.form.has_file("file")) {
            res.status = 400;
            res.set_content("{\"success\":false,\"error\":\"no file field in multipart form\"}", "application/json");
            return;
        }
        const auto& file = req.form.get_file("file");
        std::string filename = file.filename;
        if (auto pos = filename.find_last_of("/\\"); pos != std::string::npos)
            filename = filename.substr(pos + 1);
        if (filename.empty() || filename == "." || filename == ".." ||
            filename.find("..") != std::string::npos) {
            res.status = 400;
            res.set_content("{\"success\":false,\"error\":\"invalid filename\"}", "application/json");
            return;
        }
        {
            std::string ext = filename;
            auto dot = ext.rfind('.');
            ext = (dot != std::string::npos) ? ext.substr(dot) : "";
            for (auto& c : ext) c = (char)std::tolower((unsigned char)c);
            if (ext != ".wav") {
                res.status = 400;
                res.set_content("{\"success\":false,\"error\":\"only .wav files are accepted\"}", "application/json");
                return;
            }
        }
        std::filesystem::path dest = std::filesystem::path(aidir) / filename;
        std::ofstream ofs(dest, std::ios::binary);
        if (!ofs.is_open()) {
            res.status = 500;
            res.set_content("{\"success\":false,\"error\":\"cannot write to audio-in directory\"}", "application/json");
            return;
        }
        ofs.write(file.content.data(), file.content.size());
        ofs.close();
        fprintf(stderr, "[sa3-server] uploaded %s (%zu bytes) -> %s\n", filename.c_str(), file.content.size(), dest.string().c_str());
        res.set_content("{\"success\":true,\"filename\":\"" + json_escape(filename) + "\"}", "application/json");
    });

    svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(embedded_web::index_html, "text/html");
    });
    svr.Get("/app.js", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(embedded_web::app_js, "application/javascript");
    });

    fprintf(stderr, "[sa3-server] http://%s:%d  model=%s/%s  models=%s  adapters=%s  prompts=%s  audio-in=%s  (async /poll_status; frugal default)\n",
            host.c_str(), port, cfg.model_variant.c_str(), cfg.encoding.c_str(), cfg.models_dir.c_str(), adir.c_str(), pdir.c_str(), aidir.c_str());
    if (!svr.listen(host.c_str(), port)) {
        fprintf(stderr, "[sa3-server] failed to bind %s:%d\n", host.c_str(), port);
        return 1;
    }
    return 0;
}
