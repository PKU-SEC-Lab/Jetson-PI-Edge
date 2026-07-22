// State-parity + context/action split test for the Jetson-PI narrow API.
//
// Verifies the two corrections that FlashRT #148 depends on:
//   1. PI0.5 proprioceptive state is serialized into the
//      `Task: ..., State: ...;\nAction:` prompt, so varying state while
//      holding images/task fixed CHANGES the action chunk (the legacy
//      llama_set_pi0_state tensor path was a no-op for PI0.5, so
//      zeros-vs-ones used to be bit-identical).
//   2. context()/action() is a real encode/decode boundary: context() retains
//      a pending prepared batch and does NOT cache the final action; action()
//      runs the decode. One-shot: action() without context() returns
//      ACTION_NOT_READY; discard_context() between the two returns
//      ACTION_NOT_READY; context() output equals infer() output.
//
// The deterministic prompt checks run without model weights. Real-model checks
// require an explicit model kind and backend so validation never silently
// changes execution mode.
//
// Env:
//   JETSONPI_PI0_MODEL    Pi0 or PI0.5 policy GGUF
//   JETSONPI_PI0_MMPROJ   VIT mmproj GGUF
//   JETSONPI_PI0_BACKEND  "cpu" | "cuda" | "vulkan" | "opencl" | "sycl"
//   JETSONPI_PI0_MODEL_KIND  "pi0" | "pi05"
//   JETSONPI_PI0_IMG      224x224 RGB image (first view)
//   JETSONPI_PI0_WRIST    224x224 RGB image (second view)
//   JETSONPI_PI0_PROMPT   task text file
//
// Skips (returns 0) when env vars are unset.

#include "jetson_pi_pi0.h"
#include "jetson_pi_pi0_prompt.h"
#include "pi-model-detect.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, msg) do { \
    const std::string check_msg = (msg); \
    if (!(cond)) { std::printf("FAIL: %s\n", check_msg.c_str()); g_fail = 1; } \
    else { std::printf("ok  : %s\n", check_msg.c_str()); } \
} while (0)

int main() {
    const float boundary_state[] = {
        -2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f,
    };
    CHECK(jetson_pi_pi0_detail::format_pi05_openpi_prompt(
              "pick", boundary_state,
              sizeof(boundary_state) / sizeof(boundary_state[0])) ==
              "Task: pick, State: -1 0 64 128 192 255 255;\nAction: ",
          "PI0.5 prompt uses exact openpi boundary bins");
    const float short_state[] = {0.0f};
    CHECK(jetson_pi_pi0_detail::format_pi05_openpi_prompt(
              "pick", short_state, 1) ==
              "Task: pick, State: 128;\nAction: ",
          "PI0.5 short state is not padded in the prompt");
    const float zero_state_for_prompt[8] = {};
    CHECK(jetson_pi_pi0_detail::format_pi05_openpi_prompt(
              "pick", zero_state_for_prompt, 8) ==
              "Task: pick, State: 128 128 128 128 128 128 128 128;\nAction: ",
          "PI0.5 NULL-state representation is eight zero bins");
    const float padded_state_for_prompt[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    CHECK(jetson_pi_pi0_detail::format_pi05_openpi_prompt(
              "pick", padded_state_for_prompt, 10) ==
              "Task: pick, State: 128 128 128 128 128 128 128 128;\nAction: ",
          "PI0.5 formatter ignores provider padding after eight values");

    const char * model_env  = std::getenv("JETSONPI_PI0_MODEL");
    const char * mmproj_env = std::getenv("JETSONPI_PI0_MMPROJ");
    const char * img_env    = std::getenv("JETSONPI_PI0_IMG");
    const char * wrist_env  = std::getenv("JETSONPI_PI0_WRIST");
    const char * prompt_env = std::getenv("JETSONPI_PI0_PROMPT");
    const char * backend_env = std::getenv("JETSONPI_PI0_BACKEND");
    const char * model_kind_env = std::getenv("JETSONPI_PI0_MODEL_KIND");
    if (!model_env || !mmproj_env || !img_env || !wrist_env || !prompt_env ||
        !backend_env || !backend_env[0] || !model_kind_env ||
        !model_kind_env[0]) {
        std::printf("SKIP - real-model env requires MODEL/MMPROJ/BACKEND/"
                    "MODEL_KIND/IMG/WRIST/PROMPT\n");
        return g_fail;
    }
    const std::string backend = backend_env;
    const bool test_pi05 = std::strcmp(model_kind_env, "pi05") == 0;
    if (!test_pi05 && std::strcmp(model_kind_env, "pi0") != 0) {
        std::printf("FAIL: JETSONPI_PI0_MODEL_KIND must be pi0 or pi05\n");
        return 1;
    }
    const pi_model_kind expected_kind = test_pi05 ? PI_MODEL_PI05 : PI_MODEL_PI0;
    const pi_model_detect_result detected =
        pi_model_detect_gguf_pair(model_env, mmproj_env);
    CHECK(detected.kind == expected_kind,
          "detected GGUF kind matches JETSONPI_PI0_MODEL_KIND");
    if (detected.kind != expected_kind) {
        std::printf("    detected=%s expected=%s reason=%s\n",
                    pi_model_kind_name(detected.kind),
                    pi_model_kind_name(expected_kind), detected.reason.c_str());
        return 1;
    }

    int iw = 0, ih = 0, ic = 0;
    unsigned char * img = stbi_load(img_env, &iw, &ih, &ic, 3);
    CHECK(img != nullptr && iw == 224 && ih == 224, "load img 224x224");
    int ww = 0, wh = 0, wc = 0;
    unsigned char * wrist = stbi_load(wrist_env, &ww, &wh, &wc, 3);
    CHECK(wrist != nullptr && ww == 224 && wh == 224, "load wrist 224x224");
    std::ifstream prompt_file(prompt_env, std::ios::binary);
    std::string prompt((std::istreambuf_iterator<char>(prompt_file)),
                       std::istreambuf_iterator<char>());
    const bool prompt_ok = prompt_file.good() || prompt_file.eof();
    CHECK(prompt_ok && !prompt.empty(), "prompt non-empty");
    if (!prompt.empty() && prompt.back() == '\n') prompt.pop_back();
    if (!img || !wrist || !prompt_ok) {
        if (img) stbi_image_free(img);
        if (wrist) stbi_image_free(wrist);
        return g_fail;
    }

    jetson_pi_pi0_config jc{};
    jc.struct_size  = sizeof(jc);
    jc.model_path   = model_env;
    jc.mmproj_path  = mmproj_env;
    jc.backend      = backend.c_str();
    jc.n_views      = 2;
    jc.image_height = 224;
    jc.image_width  = 224;
    jc.n_threads    = 0;

    jetson_pi_pi0 * pi0 = nullptr;
    int32_t s = jetson_pi_pi0_open(&jc, &pi0);
    CHECK(s == JETSON_PI_PI0_OK && pi0, "jetson_pi_pi0_open");
    if (s != JETSON_PI_PI0_OK) {
        std::printf("FAIL: open (rc=%d): %s\n", s, jetson_pi_pi0_open_error());
        return 1;
    }

    uint32_t action_steps = 0, action_dim = 0;
    CHECK(jetson_pi_pi0_action_shape(pi0, &action_steps, &action_dim) ==
              JETSON_PI_PI0_OK && action_steps > 0 && action_dim > 0,
          "action_shape");
    const size_t n_elems = static_cast<size_t>(action_steps) * action_dim;
    const size_t state_dim = test_pi05 ? 8 : static_cast<size_t>(action_dim);
    std::printf("    action_steps=%u action_dim=%u state_dim=%zu\n",
                action_steps, action_dim, state_dim);

    const uint8_t * imgs[2] = { img, wrist };

    // Baseline: an explicit full-width zero state.
    std::vector<float> zero_state(state_dim, 0.0f);
    std::vector<float> actions_zero(n_elems, 0.0f);
    size_t written = 0;
    s = jetson_pi_pi0_infer(pi0, imgs, 2, prompt.data(), prompt.size(),
                            zero_state.data(), state_dim,
                            actions_zero.data(), n_elems, &written);
    CHECK(s == JETSON_PI_PI0_OK && written == n_elems, "infer with zero state");
    if (s != JETSON_PI_PI0_OK) {
        std::printf("FAIL: infer zero state (rc=%d): %s\n", s,
                    jetson_pi_pi0_last_error(pi0));
        jetson_pi_pi0_close(pi0);
        return 1;
    }

    // NULL follows the public C API contract and means the same zero state.
    {
        std::vector<float> actions_null(n_elems, 0.0f);
        written = 0;
        s = jetson_pi_pi0_infer(pi0, imgs, 2, prompt.data(), prompt.size(),
                                nullptr, 0,
                                actions_null.data(), n_elems, &written);
        CHECK(s == JETSON_PI_PI0_OK && written == n_elems,
              "infer with NULL state");
        float max_diff = 0.0f;
        for (size_t i = 0; i < n_elems; ++i) {
            max_diff = std::max(
                max_diff, std::fabs(actions_null[i] - actions_zero[i]));
        }
        CHECK(max_diff <= 1e-5f,
              "NULL state matches explicit zero state");
    }

    float dummy_state = 0.0f;
    s = jetson_pi_pi0_context(pi0, imgs, 2, prompt.data(), prompt.size(),
                              &dummy_state, 0);
    CHECK(s == JETSON_PI_PI0_INVALID,
          "non-NULL state with zero count is rejected");
    s = jetson_pi_pi0_context(pi0, imgs, 2, prompt.data(), prompt.size(),
                              nullptr, 1);
    CHECK(s == JETSON_PI_PI0_INVALID,
          "NULL state with nonzero count is rejected");
    const float nan_state = std::numeric_limits<float>::quiet_NaN();
    s = jetson_pi_pi0_context(pi0, imgs, 2, prompt.data(), prompt.size(),
                              &nan_state, 1);
    CHECK(s == JETSON_PI_PI0_INVALID, "non-finite state is rejected");

    if (test_pi05) {
        if (action_dim > 8) {
            std::vector<float> provider_padded(action_dim, 0.0f);
            std::vector<float> actions_padded(n_elems, 0.0f);
            written = 0;
            s = jetson_pi_pi0_infer(pi0, imgs, 2, prompt.data(), prompt.size(),
                                    provider_padded.data(), provider_padded.size(),
                                    actions_padded.data(), n_elems, &written);
            CHECK(s == JETSON_PI_PI0_OK && written == n_elems,
                  "PI0.5 accepts action_dim-wide zero-padded provider state");
            float max_diff = 0.0f;
            for (size_t i = 0; i < n_elems; ++i) {
                max_diff = std::max(max_diff,
                    std::fabs(actions_padded[i] - actions_zero[i]));
            }
            CHECK(max_diff <= 1e-5f,
                  "PI0.5 provider padding matches explicit eight-value state");

            provider_padded[8] = 1.0f;
            s = jetson_pi_pi0_context(pi0, imgs, 2, prompt.data(), prompt.size(),
                                      provider_padded.data(), provider_padded.size());
            CHECK(s == JETSON_PI_PI0_STATE_SIZE,
                  "PI0.5 rejects nonzero values in provider padding");
        }
        struct Case { const char * name; float value; };
        const Case cases[] = {
            {"x<-1 (literal -1 token)", -2.0f},
            {"x in (-1,0)",              -0.5f},
            {"x in (0,1)",                0.5f},
            {"x>=1 (bin 255)",            1.5f},
        };
        for (const Case & c : cases) {
            std::vector<float> st(state_dim, c.value);
            std::vector<float> a(n_elems, 0.0f);
            written = 0;
            s = jetson_pi_pi0_infer(pi0, imgs, 2, prompt.data(), prompt.size(),
                                    st.data(), state_dim,
                                    a.data(), n_elems, &written);
            CHECK(s == JETSON_PI_PI0_OK, std::string("infer ") + c.name);
            float max_diff = 0.0f;
            for (size_t i = 0; i < n_elems; ++i) {
                max_diff = std::max(
                    max_diff, std::fabs(a[i] - actions_zero[i]));
            }
            std::printf("    %s: max abs diff vs zero-state = %.9g\n",
                        c.name, max_diff);
            CHECK(max_diff > 0.0f,
                  std::string("state in prompt changes actions: ") + c.name);
        }
    } else {
        std::vector<float> short_legacy = {0.25f};
        std::vector<float> padded_legacy(state_dim, 0.0f);
        padded_legacy[0] = short_legacy[0];
        std::vector<float> actions_short(n_elems, 0.0f);
        std::vector<float> actions_padded(n_elems, 0.0f);
        written = 0;
        s = jetson_pi_pi0_infer(pi0, imgs, 2, prompt.data(), prompt.size(),
                                short_legacy.data(), short_legacy.size(),
                                actions_short.data(), n_elems, &written);
        CHECK(s == JETSON_PI_PI0_OK && written == n_elems,
              "legacy Pi0 accepts short state");
        written = 0;
        s = jetson_pi_pi0_infer(pi0, imgs, 2, prompt.data(), prompt.size(),
                                padded_legacy.data(), padded_legacy.size(),
                                actions_padded.data(), n_elems, &written);
        CHECK(s == JETSON_PI_PI0_OK && written == n_elems,
              "legacy Pi0 accepts action_dim state");
        float max_diff = 0.0f;
        float max_state_effect = 0.0f;
        for (size_t i = 0; i < n_elems; ++i) {
            max_diff = std::max(
                max_diff, std::fabs(actions_short[i] - actions_padded[i]));
            max_state_effect = std::max(
                max_state_effect,
                std::fabs(actions_padded[i] - actions_zero[i]));
        }
        CHECK(max_diff <= 1e-5f,
              "legacy short state matches explicit zero padding");
        CHECK(max_state_effect > 0.0f,
              "legacy tensor state changes actions");
    }

    // ---- TEST 2: context/action real split + one-shot semantics ----
    // 2a. action() without context() -> ACTION_NOT_READY
    {
        jetson_pi_pi0_discard_context(pi0);  // ensure no pending context
        std::vector<float> a(n_elems, 0.0f);
        written = 123;
        s = jetson_pi_pi0_action(pi0, a.data(), n_elems, &written);
        CHECK(s == JETSON_PI_PI0_ACTION_NOT_READY && written == 0,
              "action without context returns ACTION_NOT_READY");
    }
    // 2b. context() then discard_context() then action() -> NOT_READY
    {
        s = jetson_pi_pi0_context(pi0, imgs, 2, prompt.data(), prompt.size(),
                                  zero_state.data(), state_dim);
        CHECK(s == JETSON_PI_PI0_OK, "context() prepares pending action");
        CHECK(jetson_pi_pi0_discard_context(pi0) == JETSON_PI_PI0_OK,
              "discard_context drops pending action");
        std::vector<float> a(n_elems, 0.0f);
        written = 123;
        s = jetson_pi_pi0_action(pi0, a.data(), n_elems, &written);
        CHECK(s == JETSON_PI_PI0_ACTION_NOT_READY && written == 0,
              "action after discard returns ACTION_NOT_READY");
    }
    // 2c. context() then action() produces the same action as whole infer()
    {
        s = jetson_pi_pi0_context(pi0, imgs, 2, prompt.data(), prompt.size(),
                                  zero_state.data(), state_dim);
        CHECK(s == JETSON_PI_PI0_OK, "context() for split parity");
        std::vector<float> a_split(n_elems, 0.0f);
        written = 0;
        s = jetson_pi_pi0_action(pi0, a_split.data(), n_elems, &written);
        CHECK(s == JETSON_PI_PI0_OK && written == n_elems,
              "action() consumes pending context");
        float max_diff = 0.0f;
        for (size_t i = 0; i < n_elems; ++i)
            max_diff = std::max(max_diff,
                                std::fabs(a_split[i] - actions_zero[i]));
        std::printf("    split vs whole-infer max abs diff = %.9g\n", max_diff);
        CHECK(max_diff <= 1e-5f,
              "context/action split matches whole infer");
    }
    // 2d. action() consumed the context: a second action() -> NOT_READY
    {
        std::vector<float> a(n_elems, 0.0f);
        written = 123;
        s = jetson_pi_pi0_action(pi0, a.data(), n_elems, &written);
        CHECK(s == JETSON_PI_PI0_ACTION_NOT_READY && written == 0,
              "action consumes pending context exactly once");
    }

    // ---- TEST 3: state above the model-specific bound is rejected ----
    {
        std::vector<float> big(state_dim + 1, 0.0f);
        s = jetson_pi_pi0_infer(pi0, imgs, 2, prompt.data(), prompt.size(),
                                big.data(), big.size(),
                                actions_zero.data(), n_elems, &written);
        CHECK(s == JETSON_PI_PI0_STATE_SIZE,
              "state wider than the model-specific bound is rejected");
    }

    jetson_pi_pi0_close(pi0);
    if (img)   stbi_image_free(img);
    if (wrist) stbi_image_free(wrist);

    std::printf(g_fail ? "\n== PI0 STATE+SPLIT FAILED ==\n"
                       : "\n== PI0 STATE+SPLIT PASSED ==\n");
    return g_fail;
}
