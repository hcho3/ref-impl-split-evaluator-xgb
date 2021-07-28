#include <gtest/gtest.h>
#include <string>

bool g_verbose_flag = false;

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--verbose") {
      g_verbose_flag = true;
      break;
    }
  }
  return RUN_ALL_TESTS();
}
