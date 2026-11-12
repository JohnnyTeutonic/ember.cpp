# CTest chat_template_conformance: tools/chat_template_check cases on paperkiln's conformance file at the
# commit chat_template.h was vendored from (tools/fixtures/chat_template_cases_877fb0d0.json, paperkiln
# 877fb0d0 tests/chat_template_cases.json, sha256 7558a2b3...). Passes when the check exits 0 (no header
# or strict-history failure) and exactly the 4 request-path differences experiments/chat_v2_2026-10-27
# documents as the contract's rules remain (v1's two system messages joined on the OpenAI path; v2's empty
# system message on all three paths): a new difference, or a lost one, fails.
#   cmake -DCHECK=<chat_template_check> -DCASES=<cases.json> -P tools/chat_template_conformance.cmake
execute_process(COMMAND "${CHECK}" cases "${CASES}" OUTPUT_VARIABLE report ERROR_VARIABLE err RESULT_VARIABLE rc)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "chat_template_check cases exited ${rc}: ${err}")
endif()
if(NOT report MATCHES "\"paths_summary\": *{[^}]*\"differ\": *4[,\n ]")
  string(REGEX MATCH "\"paths_summary\": *{[^}]*}" summary "${report}")
  message(FATAL_ERROR "request-path differences are not the documented 4: ${summary}")
endif()
string(REGEX MATCH "\"paths_summary\": *{[^}]*}" summary "${report}")
message(STATUS "chat_template_check: 0 failures; ${summary}")
