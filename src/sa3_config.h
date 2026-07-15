#pragma once
// sa3_config.h — single source of truth for all config defaults.
// Every CLI tool / server reads its config from Sa3Config; no env vars, no hardcoded
// defaults in arg-parsing loops or JSON handlers. The struct is the one place to change
// a default.

#include "audio_post.h"

#include <string>

struct Sa3Config {
    // Model identity
    std::string model_variant = "medium";
    std::string encoding      = "f16";

    // Directories (models_dir is mandatory — no default)
    std::string models_dir  = "models";
    std::string adapters_dir  = "loras";
    std::string prompts_dir   = "prompts";
    std::string audio_in_dir  = "audio-in";
    std::string dump_cond_dir;                  // empty → disabled

    // Backend
    std::string device;                         // "" → auto (GPU if avail), "cpu" → CPU
    std::string gpu_selector;                   // "" → auto
    int cpu_threads          = 0;               // 0 → auto
    int flash_attn           = 0;               // 0=off, 1=on
    int same_flash_attn_mode = 0;               // 0=off, 1=full, 2=local
    int profile              = 0;               // 0=off, 1=on

    // Server bind (server only)
    std::string host = "127.0.0.1";
    int port = 8006;

    // Generation defaults
    double duration          = 30.0;
    double max_duration      = 300.0;
    int    steps             = 8;
    int    default_loop_bars = 8;
    double loop_pad_seconds  = 2.0;
    double cfg_scale         = 1.0;
    double cfg_rescale       = 0.0;
    double apg_scale         = 1.0;
    double cfg_norm_threshold = 0.0;
    double cfg_interval_min  = 0.0;
    double cfg_interval_max  = 1.0;
    double init_noise_level  = 0.85;
    double inpaint_start     = -1.0;
    double inpaint_end       = -1.0;
    long long seed           = -1;
    int    encode_chunk_size = 512;
    int    encode_overlap    = 32;
    int    decode_chunk_size = 512;
    int    decode_overlap    = 32;
    double bpm               = 120.0;

    // Loudness (LoudnessParams ctor provides its own member defaults)
    sa3::LoudnessParams loudness;
};
