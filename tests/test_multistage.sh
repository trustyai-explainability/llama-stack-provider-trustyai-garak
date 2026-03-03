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
    echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║  Multi-stage Containerfile Test Suite                 ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# Print test name
print_test() {
    echo -e "${YELLOW}▶ TEST:${NC} $1"
}

# Assert command should fail
assert_fails() {
    local description=$1
    shift
    local cmd="$*"

    print_test "$description"
    TESTS+=("$description")

    if eval "$cmd" &>/dev/null; then
        echo -e "  ${RED}✗ FAIL${NC}: Command succeeded but should have failed"
        echo -e "  ${RED}Command:${NC} $cmd"
        ((FAILED++))
        return 1
    else
        echo -e "  ${GREEN}✓ PASS${NC}: Command failed as expected"
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
        echo -e "  ${GREEN}✓ PASS${NC}: Command succeeded"
        if [[ -n "$output" ]]; then
            echo -e "  ${BLUE}Output:${NC} $output"
        fi
        ((PASSED++))
        return 0
    else
        echo -e "  ${RED}✗ FAIL${NC}: Command failed but should have succeeded"
        echo -e "  ${RED}Command:${NC} $cmd"
        echo -e "  ${RED}Error:${NC} $output"
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
            echo -e "  ${GREEN}✓ PASS${NC}: Output matches expected value"
            echo -e "  ${BLUE}Expected:${NC} $expected"
            echo -e "  ${BLUE}Got:${NC} $output"
            ((PASSED++))
            return 0
        else
            echo -e "  ${RED}✗ FAIL${NC}: Output does not match expected value"
            echo -e "  ${BLUE}Expected:${NC} $expected"
            echo -e "  ${RED}Got:${NC} $output"
            ((FAILED++))
            return 1
        fi
    else
        echo -e "  ${RED}✗ FAIL${NC}: Command failed"
        echo -e "  ${RED}Command:${NC} $cmd"
        echo -e "  ${RED}Error:${NC} $output"
        ((FAILED++))
        return 1
    fi
}

# Print summary
print_summary() {
    echo ""
    echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}Test Summary${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
    echo -e "Total tests: $((PASSED + FAILED))"
    echo -e "${GREEN}Passed: $PASSED${NC}"

    if [[ $FAILED -gt 0 ]]; then
        echo -e "${RED}Failed: $FAILED${NC}"
        echo ""
        echo -e "${RED}Failed tests:${NC}"
        # This would require tracking which tests failed, keeping it simple for now
    fi

    echo ""
    if [[ $FAILED -eq 0 ]]; then
        echo -e "${GREEN}✓ All tests passed!${NC}"
        return 0
    else
        echo -e "${RED}✗ Some tests failed${NC}"
        return 1
    fi
}

# Main test execution
main() {
    print_header

    # Build the image
    echo -e "${YELLOW}Building image...${NC}"
    if ! podman build -f Containerfile -t garak-evalhub-local . > /tmp/build.log 2>&1; then
        echo -e "${RED}✗ Build failed!${NC}"
        echo "See /tmp/build.log for details"
        exit 1
    fi
    echo -e "${GREEN}✓ Build successful${NC}"
    echo ""

    # Test 1-3: Build tools should NOT be in runtime image
    assert_fails "patchelf is not in runtime image" \
        "podman run --rm garak-evalhub-local which patchelf"

    assert_fails "cargo is not in runtime image" \
        "podman run --rm garak-evalhub-local which cargo"

    assert_fails "rustc is not in runtime image" \
        "podman run --rm garak-evalhub-local which rustc"

    # Test 4: Package should be importable
    assert_succeeds "Python package imports successfully" \
        "podman run --rm garak-evalhub-local python -c 'import llama_stack_provider_trustyai_garak; print(\"OK\")'"

    # Test 5: Using correct Python (venv)
    assert_equals "Python is using venv at /opt/app-root/bin/python" \
        "/opt/app-root/bin/python" \
        "podman run --rm garak-evalhub-local python -c 'import sys; print(sys.executable)'"

    # Test 6: Verify non-root user
    assert_equals "Container runs as user 1001" \
        "1001" \
        "podman run --rm garak-evalhub-local id -u"

    # Test 7: Verify Python works without explicit path
    assert_succeeds "Plain 'python' command works" \
        "podman run --rm garak-evalhub-local python --version"

    # Print summary and exit with appropriate code
    echo ""
    print_summary
}

main "$@"
