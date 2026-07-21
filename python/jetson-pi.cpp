#include "jetson_pi_pi0.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

class pi_model {
public:
    pi_model(
            std::string model_path,
            std::string mmproj_path,
            std::string backend,
            uint32_t n_views,
            uint32_t image_height,
            uint32_t image_width,
            int32_t n_threads) :
        model_path(std::move(model_path)),
        mmproj_path(std::move(mmproj_path)),
        backend(std::move(backend)),
        n_views(n_views),
        image_height(image_height),
        image_width(image_width) {
        jetson_pi_pi0_config config = {};
        config.struct_size  = sizeof(config);
        config.model_path   = this->model_path.c_str();
        config.mmproj_path  = this->mmproj_path.c_str();
        config.backend      = this->backend.c_str();
        config.n_views      = this->n_views;
        config.image_height = this->image_height;
        config.image_width  = this->image_width;
        config.n_threads    = n_threads;

        jetson_pi_pi0 * opened = nullptr;
        int32_t status;
        {
            py::gil_scoped_release release;
            status = jetson_pi_pi0_open(&config, &opened);
        }
        if (status != JETSON_PI_PI0_OK) {
            throw std::runtime_error(std::string("jetson_pi.load_model: ") + jetson_pi_pi0_open_error());
        }
        handle = opened;

        status = jetson_pi_pi0_action_shape(handle, &action_steps, &action_dim);
        if (status != JETSON_PI_PI0_OK) {
            const std::string message = jetson_pi_pi0_last_error(handle);
            jetson_pi_pi0_close(handle);
            handle = nullptr;
            throw std::runtime_error("jetson_pi.load_model: could not read action shape: " + message);
        }
    }

    ~pi_model() {
        py::gil_scoped_release release;
        close_noexcept();
    }

    pi_model(const pi_model &) = delete;
    pi_model & operator=(const pi_model &) = delete;

    py::array_t<float> predict(
            const py::array & images,
            const std::string & prompt,
            const py::object & state) {
        if (!images.dtype().is(py::dtype::of<uint8_t>())) {
            throw py::type_error("jetson_pi.PIModel.predict: images must have dtype uint8");
        }
        if ((images.flags() & py::array::c_style) == 0) {
            throw py::value_error("jetson_pi.PIModel.predict: images must be C-contiguous");
        }
        if (images.ndim() != 4) {
            throw py::value_error("jetson_pi.PIModel.predict: images must have shape [n_views, height, width, 3]");
        }
        if (images.shape(0) != static_cast<py::ssize_t>(n_views) ||
            images.shape(1) != static_cast<py::ssize_t>(image_height) ||
            images.shape(2) != static_cast<py::ssize_t>(image_width) ||
            images.shape(3) != 3) {
            throw py::value_error(
                "jetson_pi.PIModel.predict: expected images shape [" +
                std::to_string(n_views) + ", " +
                std::to_string(image_height) + ", " +
                std::to_string(image_width) + ", 3]");
        }
        if (prompt.empty()) {
            throw py::value_error("jetson_pi.PIModel.predict: prompt must not be empty");
        }

        const float * state_data = nullptr;
        size_t state_size = 0;
        py::array state_array;
        if (!state.is_none()) {
            if (!py::isinstance<py::array>(state)) {
                throw py::type_error("jetson_pi.PIModel.predict: state must be a NumPy array or None");
            }
            state_array = py::reinterpret_borrow<py::array>(state);
            if (!state_array.dtype().is(py::dtype::of<float>())) {
                throw py::type_error("jetson_pi.PIModel.predict: state must have dtype float32");
            }
            if ((state_array.flags() & py::array::c_style) == 0) {
                throw py::value_error("jetson_pi.PIModel.predict: state must be C-contiguous");
            }
            if (state_array.ndim() != 1) {
                throw py::value_error("jetson_pi.PIModel.predict: state must have shape [state_dim]");
            }
            state_data = static_cast<const float *>(state_array.data());
            state_size = static_cast<size_t>(state_array.shape(0));
        }

        const auto * image_data = static_cast<const uint8_t *>(images.data());
        const size_t image_stride = static_cast<size_t>(image_height) * image_width * 3;
        std::vector<const uint8_t *> image_ptrs(n_views);
        for (uint32_t i = 0; i < n_views; ++i) {
            image_ptrs[i] = image_data + static_cast<size_t>(i) * image_stride;
        }

        py::array_t<float> actions({
            static_cast<py::ssize_t>(action_steps),
            static_cast<py::ssize_t>(action_dim),
        });
        float * actions_data = actions.mutable_data();
        const size_t actions_size = static_cast<size_t>(actions.size());
        size_t actions_written = 0;
        int32_t status;
        std::string error_message;
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex);
            ensure_open();
            status = jetson_pi_pi0_infer(
                handle,
                image_ptrs.data(),
                n_views,
                prompt.data(),
                prompt.size(),
                state_data,
                state_size,
                actions_data,
                actions_size,
                &actions_written);
            if (status != JETSON_PI_PI0_OK) {
                error_message = jetson_pi_pi0_last_error(handle);
            }
        }
        if (status != JETSON_PI_PI0_OK) {
            raise_infer_error(status, error_message);
        }
        if (actions_written != actions_size) {
            throw std::runtime_error("jetson_pi.PIModel.predict: inference returned an invalid action size");
        }
        return actions;
    }

    void close() {
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(mutex);
        if (handle != nullptr) {
            jetson_pi_pi0_close(handle);
            handle = nullptr;
        }
    }

    pi_model & enter() {
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex);
            ensure_open();
        }
        return *this;
    }

    void exit(const py::object &, const py::object &, const py::object &) {
        close();
    }

    std::pair<uint32_t, uint32_t> get_action_shape() const {
        return { action_steps, action_dim };
    }

    std::pair<uint32_t, uint32_t> get_image_size() const {
        return { image_height, image_width };
    }

    uint32_t get_n_views() const {
        return n_views;
    }

    const std::string & get_backend() const {
        return backend;
    }

private:
    void ensure_open() const {
        if (handle == nullptr) {
            throw std::runtime_error("jetson_pi.PIModel: model is closed");
        }
    }

    [[noreturn]] void raise_infer_error(int32_t status, const std::string & detail) const {
        const std::string message = "jetson_pi.PIModel.predict: " + detail;
        if (status == JETSON_PI_PI0_INVALID ||
            status == JETSON_PI_PI0_DIM_MISMATCH ||
            status == JETSON_PI_PI0_STATE_SIZE) {
            throw py::value_error(message);
        }
        throw std::runtime_error(message);
    }

    void close_noexcept() noexcept {
        std::lock_guard<std::mutex> lock(mutex);
        if (handle != nullptr) {
            jetson_pi_pi0_close(handle);
            handle = nullptr;
        }
    }

    jetson_pi_pi0 * handle = nullptr;
    std::string model_path;
    std::string mmproj_path;
    std::string backend;
    uint32_t n_views;
    uint32_t image_height;
    uint32_t image_width;
    uint32_t action_steps = 0;
    uint32_t action_dim = 0;
    mutable std::mutex mutex;
};

} // namespace

PYBIND11_MODULE(jetson_pi, module) {
    module.doc() = "NumPy inference API for Jetson-PI PI0 and PI0.5 policies";

    py::class_<pi_model>(module, "PIModel")
        .def("predict", &pi_model::predict,
            py::arg("images"), py::arg("prompt"), py::arg("state") = py::none())
        .def("infer", &pi_model::predict,
            py::arg("images"), py::arg("prompt"), py::arg("state") = py::none())
        .def("close", &pi_model::close)
        .def("__enter__", &pi_model::enter, py::return_value_policy::reference_internal)
        .def("__exit__", &pi_model::exit)
        .def_property_readonly("action_shape", &pi_model::get_action_shape)
        .def_property_readonly("image_size", &pi_model::get_image_size)
        .def_property_readonly("n_views", &pi_model::get_n_views)
        .def_property_readonly("backend", &pi_model::get_backend);

    module.def("load_model",
        [](const std::string & model_path,
           const std::string & mmproj_path,
           const std::string & backend,
           uint32_t n_views,
           uint32_t image_height,
           uint32_t image_width,
           int32_t n_threads) {
            return new pi_model(
                model_path,
                mmproj_path,
                backend,
                n_views,
                image_height,
                image_width,
                n_threads);
        },
        py::arg("model_path"),
        py::arg("mmproj_path"),
        py::arg("backend") = "cuda",
        py::arg("n_views") = 2,
        py::arg("image_height") = 224,
        py::arg("image_width") = 224,
        py::arg("n_threads") = 0,
        py::return_value_policy::take_ownership);
}
