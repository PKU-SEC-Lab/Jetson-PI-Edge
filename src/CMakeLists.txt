llama_add_compile_flags()

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/pi0-build-config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/pi0-build-config.h
    @ONLY
)

#
# libraries
#

# llama

file(GLOB LLAMA_MODELS_SOURCES "models/*.cpp")

add_library(llama
            ../include/llama.h
            llama.cpp
            llama-adapter.cpp
            llama-arch.cpp
            llama-batch.cpp
            llama-chat.cpp
            llama-context.cpp
            llama-cparams.cpp
            llama-grammar.cpp
            llama-graph.cpp
            llama-hparams.cpp
            llama-impl.cpp
            llama-io.cpp
            llama-kv-cache.cpp
            llama-kv-cache-iswa.cpp
            llama-kv-cache-dsa.cpp
            llama-kv-cache-dsv4.cpp
            llama-memory.cpp
            llama-memory-hybrid.cpp
            llama-memory-hybrid-iswa.cpp
            llama-memory-recurrent.cpp
            llama-mmap.cpp
            llama-model-loader.cpp
            llama-model-saver.cpp
            llama-model.cpp
            llama-quant.cpp
            llama-sampler.cpp
            llama-vocab.cpp
            unicode-data.cpp
            unicode.cpp
            unicode.h
            ${LLAMA_MODELS_SOURCES}
            )

set_target_properties(llama PROPERTIES
    VERSION ${LLAMA_INSTALL_VERSION}
    SOVERSION 0
    MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
)

target_include_directories(llama PRIVATE .)
target_include_directories(llama PUBLIC ../include)
target_compile_features   (llama PRIVATE cxx_std_17) # don't bump

target_link_libraries(llama PUBLIC ggml)

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_BUILD)
    target_compile_definitions(llama PUBLIC  LLAMA_SHARED)
endif()

if (LLAMA_BUILD_COMMON AND (LLAMA_BUILD_TOOLS OR JETSON_PI_BUILD_MTMD))
    add_library(jetson_pi_pi0
        ../include/jetson_pi_pi0.h
        jetson_pi_pi0.cpp
    )
    add_library(jetson_pi_llm
        ../include/jetson_pi_llm.h
        jetson_pi_llm.cpp
    )
    add_library(jetson_pi_mllm
        ../include/jetson_pi_mllm.h
        jetson_pi_mllm.cpp
    )

    foreach(target jetson_pi_pi0 jetson_pi_llm jetson_pi_mllm)
        set_target_properties(${target} PROPERTIES
            VERSION ${LLAMA_INSTALL_VERSION}
            SOVERSION 0
            INSTALL_RPATH "$ORIGIN"
        )
        target_include_directories(${target} PUBLIC ../include)
        target_compile_features(${target} PRIVATE cxx_std_17)
        if (NOT BUILD_SHARED_LIBS)
            set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
        endif()
    endforeach()

    target_link_libraries(jetson_pi_pi0 PUBLIC mtmd llama ggml)
    target_link_libraries(jetson_pi_llm PUBLIC llama ggml)
    target_link_libraries(jetson_pi_mllm PUBLIC mtmd llama ggml)

    install(TARGETS jetson_pi_pi0 jetson_pi_llm jetson_pi_mllm
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
    install(FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/../include/jetson_pi_pi0.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../include/jetson_pi_llm.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../include/jetson_pi_mllm.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
endif()
