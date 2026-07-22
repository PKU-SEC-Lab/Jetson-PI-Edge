// PI0.5 state-parity + context/action split test for the Jetson-PI narrow API.
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
// State is passed already normalized to [-1,1] (caller contract, matching the
// foreground server which bins pi0_req.state directly). Boundary regions are
// exercised explicitly because the openpi discretizer maps x<-1 to a literal
// -1 token distinct from bin 0.
//
// Env:
//   JETSONPI_PI0_MODEL    PI0.5 policy GGUF
//   JETSONPI_PI0_MMPROJ   VIT mmproj GGUF
//   JETSONPI_PI0_BACKEND  "cpu" | "cuda" | "vulkan" | "opencl" | "sycl"
//   JETSONPI_PI0_IMG      224x224 RGB image (first view)
//   JETSONPI_PI0_WRIST    224x224 RGB image (second view)
//   JETSONPI_PI0_PROMPT   task text file
//
// Skips (returns 0) when env vars are unset.

#include "jetson_pi_pi0.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::printf("FAIL: %s\n", (msg)); g_fail = 1; } \
    else { std::printf("ok  : %s\n", (msg)); } \
} while (0)

static std::string read_file(const std::string & path, bool * ok) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { if (ok) *ok = false; return {}; }
    std::string s((std::istreambuf_iterator<char>(f)),
                  std::istreambuf_iterator<char>());
    if (ok) *ok = true;
    return s;
}

int main() {
    const char * model_env  = std::getenv("JETSONPI_PI0_MODEL");
    const char * mmproj_env = std::getenv("JETSONPI_PI0_MMPROJ");
    const char * img_env    = std::getenv("JETSONPI_PI0_IMG");
    const char * wrist_env  = std::getenv("JETSONPI_PI0_WRIST");
    const char * prompt_env = std::getenv("JETSONPI_PI0_PROMPT");
    const char * backend_env = std::getenv("JETSONPI_PI0_BACKEND");
    if (!model_env || !mmproj_env || !img_env || !wrist_env || !prompt_env) {
        std::printf("SKIP - JETSONPI_PI0_MODEL/MMPROJ/IMG/WRIST/PROMPT not set\n");
        return 0;
    }
    const std::string backend = backend_env ? backend_env : "cpu";

    int iw = 0, ih = 0, ic = 0;
    unsigned char * img = stbi_load(img_env, &iw, &ih, &ic, 3);
    CHECK(img != nullptr && iw == 224 && ih == 224, "load img 224x224");
    int ww = 0, wh = 0, wc = 0;
    unsigned char * wrist = stbi_load(wrist_env, &ww, &wh, &wc, 3);
    CHECK(wrist != nullptr && ww == 224 && wh == 224, "load wrist 224x224");
    bool prompt_ok = false;
    std::string prompt = read_file(prompt_env, &prompt_ok);
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
    // PI0.5 proprioception width is 8 (openpi libero), independent of action_dim.
    const size_t state_dim = 8;
    std::printf("    action_steps=%u action_dim=%u state_dim=%zu\n",
                action_steps, action_dim, state_dim);

    const uint8_t * imgs[2] = { img, wrist };

    // Baseline: zero state (all bins in the -1..0 region).
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

    // ---- TEST 1: state in prompt changes actions (boundary regions) ----
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
        for (size_t i = 0; i < n_elems; ++i)
            max_diff = std::max(max_diff, std::fabs(a[i] - actions_zero[i]));
        std::printf("    %s: max abs diff vs zero-state = %.9g\n",
                    c.name, max_diff);
        CHECK(max_diff > 0.0f,
              std::string("state in prompt changes actions: ") + c.name);
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

    // ---- TEST 3: state size > 8 rejected for PI0.5 ----
    {
        std::vector<float> big(state_dim + 1, 0.0f);
        s = jetson_pi_pi0_infer(pi0, imgs, 2, prompt.data(), prompt.size(),
                                big.data(), big.size(),
                                actions_zero.data(), n_elems, &written);
        CHECK(s == JETSON_PI_PI0_STATE_SIZE,
              "PI0.5 rejects state wider than proprioception (8)");
    }

    jetson_pi_pi0_close(pi0);
    if (img)   stbi_image_free(img);
    if (wrist) stbi_image_free(wrist);

    std::printf(g_fail ? "\n== PI0.5 STATE+SPLIT FAILED ==\n"
                       : "\n== PI0.5 STATE+SPLIT PASSED ==\n");
    return g_fail;
}
