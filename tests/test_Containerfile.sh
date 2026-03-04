#!/bin/bash
set -euo pipefail

# Multi-stage Containerfile Test Suite
# Tests that build tools are excluded from runtime and app works correctly

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counters
PASSED=0
FAILED=0
TESTS=()

# Print test header
print_header() {
    printf "${BLUE}╔════════════════════════════════════════════════════════╗${NC}\n"
    printf "${BLUE}║  Multi-stage Containerfile Test Suite                  ║${NC}\n"
    printf "${BLUE}╚════════════════════════════════════════════════════════╝${NC}\n"
    printf "\n"
}

# Print test name
print_test() {
    printf "${YELLOW}▶ TEST:${NC} %s\n" "$1"
}

# Assert command should fail
assert_fails() {
    local description=$1
    shift
    local cmd="$*"

    print_test "$description"
    TESTS+=("$description")

    if eval "$cmd" &>/dev/null; then
        printf "  ${RED}✗ FAIL${NC}: Command succeeded but should have failed\n"
        printf "  ${RED}Command:${NC} %s\n" "$cmd"
        ((FAILED++))
        return 1
    else
        printf "  ${GREEN}✓ PASS${NC}: Command failed as expected\n"
        ((PASSED++))
        return 0
    fi
}

# Assert command should succeed
assert_succeeds() {
    local description=$1
    shift
    local cmd="$*"

    print_test "$description"
    TESTS+=("$description")

    if output=$(eval "$cmd" 2>&1); then
        printf "  ${GREEN}✓ PASS${NC}: Command succeeded\n"
        if [[ -n "$output" ]]; then
            printf "  ${BLUE}Output:${NC} %s\n" "$output"
        fi
        ((PASSED++))
        return 0
    else
        printf "  ${RED}✗ FAIL${NC}: Command failed but should have succeeded\n"
        printf "  ${RED}Command:${NC} %s\n" "$cmd"
        printf "  ${RED}Error:${NC} %s\n" "$output"
        ((FAILED++))
        return 1
    fi
}

# Assert command output equals expected value
assert_equals() {
    local description=$1
    local expected=$2
    shift 2
    local cmd="$*"

    print_test "$description"
    TESTS+=("$description")

    if output=$(eval "$cmd" 2>&1); then
        if [[ "$output" == "$expected" ]]; then
            printf "  ${GREEN}✓ PASS${NC}: Output matches expected value\n"
            printf "  ${BLUE}Expected:${NC} %s\n" "$expected"
            printf "  ${BLUE}Got:${NC} %s\n" "$output"
            ((PASSED++))
            return 0
        else
            printf "  ${RED}✗ FAIL${NC}: Output does not match expected value\n"
            printf "  ${BLUE}Expected:${NC} %s\n" "$expected"
            printf "  ${RED}Got:${NC} %s\n" "$output"
            ((FAILED++))
            return 1
        fi
    else
        printf "  ${RED}✗ FAIL${NC}: Command failed\n"
        printf "  ${RED}Command:${NC} %s\n" "$cmd"
        printf "  ${RED}Error:${NC} %s\n" "$output"
        ((FAILED++))
        return 1
    fi
}

# Print summary
print_summary() {
    printf "\n"
    printf "${BLUE}════════════════════════════════════════════════════════${NC}\n"
    printf "${BLUE}Test Summary${NC}\n"
    printf "${BLUE}════════════════════════════════════════════════════════${NC}\n"
    printf "Total tests: %d\n" "$((PASSED + FAILED))"
    printf "${GREEN}Passed: %d${NC}\n" "$PASSED"

    if [[ $FAILED -gt 0 ]]; then
        printf "${RED}Failed: %d${NC}\n" "$FAILED"
        printf "\n"
        printf "${RED}Failed tests:${NC}\n"
        # This would require tracking which tests failed, keeping it simple for now
    fi

    printf "\n"
    if [[ $FAILED -eq 0 ]]; then
        printf "${GREEN}✓ All tests passed!${NC}\n"
        return 0
    else
        printf "${RED}✗ Some tests failed${NC}\n"
        return 1
    fi
}

# Main test execution
main() {
    print_header

    # Build the image
    printf "${YELLOW}Building image...${NC}\n"
    if ! podman build -f Containerfile -t garak-evalhub-local .; then
        printf "${RED}✗ Build failed!${NC}\n"
        exit 1
    fi
    printf "${GREEN}✓ Build successful${NC}\n"
    printf "\n"

    # Test: Build tools should NOT be in runtime image
    assert_fails "patchelf is not in runtime image" \
        "podman run --rm garak-evalhub-local which patchelf"

    assert_fails "cargo is not in runtime image" \
        "podman run --rm garak-evalhub-local which cargo"

    assert_fails "rustc is not in runtime image" \
        "podman run --rm garak-evalhub-local which rustc"

    # Test: Package should be importable
    assert_succeeds "Python package imports successfully" \
        "podman run --rm garak-evalhub-local python -c 'import llama_stack_provider_trustyai_garak; print(\"OK\")'"

    # Test: Using correct Python (venv)
    assert_equals "Python is using venv at /opt/app-root/bin/python" \
        "/opt/app-root/bin/python" \
        "podman run --rm garak-evalhub-local python -c 'import sys; print(sys.executable)'"

    # Test: Verify non-root user
    assert_equals "Container runs as user 1001" \
        "1001" \
        "podman run --rm garak-evalhub-local id -u"

    # Test: Verify Python works without explicit path
    assert_succeeds "Plain 'python' command works" \
        "podman run --rm garak-evalhub-local python --version"

    # Test: Package version is accessible
    assert_succeeds "Package version is accessible" \
        "podman run --rm garak-evalhub-local python -c 'import llama_stack_provider_trustyai_garak; print(llama_stack_provider_trustyai_garak.__version__)'"

    # Test: Image has version label
    assert_succeeds "Image has org.opencontainers.image.version label" \
        "podman inspect garak-evalhub-local | jq -r '.[0].Config.Labels.\"org.opencontainers.image.version\"'"

    # Test: .git directory is NOT in final image
    assert_fails ".git directory is not in final image" \
        "podman run --rm garak-evalhub-local ls -la / | grep -q '\\.git'"

    # Print summary and exit with appropriate code
    print_summary
}

main "$@"
