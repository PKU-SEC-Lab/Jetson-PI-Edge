find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    RESULT_VARIABLE JETSON_PI_PYBIND11_RESULT
    OUTPUT_VARIABLE JETSON_PI_PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if (JETSON_PI_PYBIND11_RESULT EQUAL 0)
    list(APPEND CMAKE_PREFIX_PATH "${JETSON_PI_PYBIND11_CMAKE_DIR}")
endif()

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(jetson_pi MODULE jetson-pi.cpp)
target_compile_features(jetson_pi PRIVATE cxx_std_17)
target_link_libraries(jetson_pi PRIVATE jetson_pi_pi0)

set_target_properties(jetson_pi PROPERTIES
    INSTALL_RPATH "$ORIGIN"
)

install(TARGETS jetson_pi
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES jetson_pi_foreground.py
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
