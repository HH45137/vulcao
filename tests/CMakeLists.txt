add_executable(vulcao_tests
    barrier_test.cpp
    headless_test.cpp
    reflection_test.cpp
    specialization_test.cpp
    vertex_layout_test.cpp
)

target_link_libraries(vulcao_tests PRIVATE
    vulcao
    doctest::doctest_with_main
)

add_test(NAME vulcao_tests COMMAND vulcao_tests)
