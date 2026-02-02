#!/bin/bash
#
# Regenerate baseline datasets for trusted players and cheaters
#
# Usage:
#   ./scripts/regenerate_baselines.sh           # Regenerate both baselines
#   ./scripts/regenerate_baselines.sh trusted   # Only trusted baseline
#   ./scripts/regenerate_baselines.sh cheaters  # Only cheater baseline
#   ./scripts/regenerate_baselines.sh --help    # Show help
#
# Options:
#   --skip-validation    Skip opponent validation (faster but no ban checking)
#   --parallel           Run both baselines in parallel (uses more API requests)
#   --include-expanded   Also build expanded-cheaters dataset after baselines
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default options
SKIP_VALIDATION=""
RUN_PARALLEL=false
RUN_TRUSTED=true
RUN_CHEATERS=true
RUN_EXPANDED=false

print_usage() {
    echo "Usage: $0 [OPTIONS] [trusted|cheaters]"
    echo ""
    echo "Regenerate baseline datasets for fairness analysis."
    echo ""
    echo "Arguments:"
    echo "  trusted     Only regenerate trusted player baseline"
    echo "  cheaters    Only regenerate cheater baseline"
    echo "  (none)      Regenerate both baselines (default)"
    echo ""
    echo "Options:"
    echo "  --skip-validation    Skip opponent ban validation (faster)"
    echo "  --parallel           Run both baselines in parallel"
    echo "  --include-expanded   Also build expanded-cheaters dataset"
    echo "  --help, -h           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                           # Regenerate both baselines"
    echo "  $0 trusted                   # Only trusted baseline"
    echo "  $0 cheaters --skip-validation  # Cheaters only, skip validation"
    echo "  $0 --parallel                # Both baselines in parallel"
    echo "  $0 --include-expanded        # Baselines + expanded cheaters"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-validation)
            SKIP_VALIDATION="--skip-opponent-validation"
            shift
            ;;
        --parallel)
            RUN_PARALLEL=true
            shift
            ;;
        --include-expanded)
            RUN_EXPANDED=true
            shift
            ;;
        --help|-h)
            print_usage
            exit 0
            ;;
        trusted)
            RUN_TRUSTED=true
            RUN_CHEATERS=false
            shift
            ;;
        cheaters)
            RUN_TRUSTED=false
            RUN_CHEATERS=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

# Check Python virtual environment
if [ ! -f "$VENV_PYTHON" ]; then
    echo -e "${RED}Error: Python virtual environment not found at $VENV_PYTHON${NC}"
    echo "Please create the virtual environment first:"
    echo "  python3 -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

cd "$PROJECT_ROOT"

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}Baseline Regeneration Script${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""
echo "Project root: $PROJECT_ROOT"
echo "Python: $VENV_PYTHON"
if [ -n "$SKIP_VALIDATION" ]; then
    echo -e "${YELLOW}Opponent validation: SKIPPED${NC}"
else
    echo "Opponent validation: Enabled"
fi
echo ""

regenerate_trusted() {
    echo -e "${GREEN}============================================================${NC}"
    echo -e "${GREEN}Regenerating TRUSTED player baseline${NC}"
    echo -e "${GREEN}============================================================${NC}"
    echo ""

    CONFIG_FILE="data/trusted/config.json"
    OUTPUT_DIR="data/trusted"

    if [ ! -f "$CONFIG_FILE" ]; then
        echo -e "${RED}Error: Config file not found: $CONFIG_FILE${NC}"
        return 1
    fi

    echo "Config: $CONFIG_FILE"
    echo "Output: $OUTPUT_DIR"
    echo ""

    "$VENV_PYTHON" scripts/generate_baseline.py \
        --config "$CONFIG_FILE" \
        --output "$OUTPUT_DIR" \
        $SKIP_VALIDATION

    echo ""
    echo -e "${GREEN}Trusted baseline regeneration complete!${NC}"
}

regenerate_cheaters() {
    echo -e "${YELLOW}============================================================${NC}"
    echo -e "${YELLOW}Regenerating CHEATER baseline${NC}"
    echo -e "${YELLOW}============================================================${NC}"
    echo ""

    CONFIG_FILE="data/cheaters/config.json"
    OUTPUT_DIR="data/cheaters"

    if [ ! -f "$CONFIG_FILE" ]; then
        echo -e "${RED}Error: Config file not found: $CONFIG_FILE${NC}"
        return 1
    fi

    echo "Config: $CONFIG_FILE"
    echo "Output: $OUTPUT_DIR"
    echo ""

    "$VENV_PYTHON" scripts/generate_baseline.py \
        --config "$CONFIG_FILE" \
        --output "$OUTPUT_DIR" \
        $SKIP_VALIDATION

    echo ""
    echo -e "${YELLOW}Cheater baseline regeneration complete!${NC}"
}

build_expanded_cheaters() {
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}Building EXPANDED CHEATERS dataset${NC}"
    echo -e "${BLUE}============================================================${NC}"
    echo ""

    echo "Output: data/expanded-cheaters/"
    echo ""

    "$VENV_PYTHON" scripts/build_expanded_cheaters.py --refresh-missing

    echo ""
    echo -e "${BLUE}Expanded cheaters dataset complete!${NC}"
}

# Run the appropriate baselines
if $RUN_PARALLEL && $RUN_TRUSTED && $RUN_CHEATERS; then
    echo -e "${BLUE}Running both baselines in parallel...${NC}"
    echo ""

    # Run in background and capture PIDs
    regenerate_trusted &
    TRUSTED_PID=$!

    regenerate_cheaters &
    CHEATERS_PID=$!

    # Wait for both to complete
    TRUSTED_STATUS=0
    CHEATERS_STATUS=0

    wait $TRUSTED_PID || TRUSTED_STATUS=$?
    wait $CHEATERS_PID || CHEATERS_STATUS=$?

    echo ""
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}SUMMARY${NC}"
    echo -e "${BLUE}============================================================${NC}"

    if [ $TRUSTED_STATUS -eq 0 ]; then
        echo -e "  Trusted baseline: ${GREEN}SUCCESS${NC}"
    else
        echo -e "  Trusted baseline: ${RED}FAILED${NC}"
    fi

    if [ $CHEATERS_STATUS -eq 0 ]; then
        echo -e "  Cheater baseline: ${GREEN}SUCCESS${NC}"
    else
        echo -e "  Cheater baseline: ${RED}FAILED${NC}"
    fi

    # Exit with error if either failed
    if [ $TRUSTED_STATUS -ne 0 ] || [ $CHEATERS_STATUS -ne 0 ]; then
        exit 1
    fi
else
    # Run sequentially
    if $RUN_TRUSTED; then
        regenerate_trusted
    fi

    if $RUN_CHEATERS; then
        regenerate_cheaters
    fi
fi

# Build expanded cheaters if requested
if $RUN_EXPANDED; then
    build_expanded_cheaters
fi

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}All baseline regeneration complete!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "Output locations:"
if $RUN_TRUSTED; then
    echo "  Trusted: data/trusted/"
fi
if $RUN_CHEATERS; then
    echo "  Cheaters: data/cheaters/"
fi
if $RUN_EXPANDED; then
    echo "  Expanded cheaters: data/expanded-cheaters/"
fi
echo ""
