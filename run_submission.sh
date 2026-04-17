#!/bin/bash
# ============================================================
# IWSLT 2026 Submission Pipeline - Team: afrinlp
# ============================================================
set -euo pipefail

TEAM_NAME="afrinlp"
AUDIO_URL="https://machinetranslation.io/files/iwslt26cloning/audio.zip"
TEXT_URL="https://machinetranslation.io/files/iwslt26cloning/text.zip"

echo "============================================================"
echo "  🚀 STARTING SUBMISSION PIPELINE FOR $TEAM_NAME"
echo "============================================================"

# 1. Setup Data Folder
mkdir -p blind_test/audio blind_test/text
echo "📥 Downloading blind test data..."
wget -q --show-progress -O audio.zip "$AUDIO_URL"
wget -q --show-progress -O text.zip "$TEXT_URL"

echo "📂 Extracting data using Python..."
python3 -m zipfile -e audio.zip blind_test/audio
python3 -m zipfile -e text.zip blind_test/text


# 2. Setup Environment
echo "🔧 Setting up OmniVoice environment..."
# Locate and patch omnivoice to use 'eager' instead of 'flex_attention' (for older torch versions)
python3 <<EOF
import omnivoice, os
try:
    p = os.path.join(os.path.dirname(omnivoice.__file__), 'model/omnivoice_llm.py')
    if os.path.exists(p):
        with open(p, 'r') as f: content = f.read()
        with open(p, 'w') as f: f.write(content.replace('flex_attention', 'eager'))
        print(f'✅ Patched {p}')
    else:
        print('⚠ OmniVoice source for patch not found, skipping.')
except Exception as e:
    print(f'⚠ Patch failed: {e}')
EOF


# 3. Generate Submission
echo "🎙️ Starting Inference (this will take 1-2 hours)..."
export TORCHDYNAMO_DISABLE=1
# Use ${PYTHONPATH:-} to avoid unbound variable error
export PYTHONPATH="./OmniVoice:${PYTHONPATH:-}"
python3 generate_submission.py --lang all --output-dir ./temp_submission


# 4. Verification
echo "🔍 Running Validator..."
for LANG in zh fr ar; do
    echo "  Validating $LANG..."
    python3 verify_submission_naming.py ./temp_submission/$LANG \
        --language "$LANG" \
        --source-file "./blind_test/text/$LANG.txt" \
        --reference-dir "./blind_test/audio/$LANG"
done

# 5. Final Packaging
echo "📦 Creating final ZIP files using Python..."
mkdir -p final_submission
for LANG in zh fr ar; do
    echo "  Zipping $LANG..."
    # Create the zip using python3 -m zipfile
    # We use a subshell to ensure we capture the directory correctly
    (cd temp_submission/$LANG && python3 -c "import zipfile, os; z = zipfile.ZipFile('../../final_submission/${TEAM_NAME}_${LANG}.zip', 'w', zipfile.ZIP_DEFLATED); [z.write(f) for f in os.listdir('.')]; z.close()")
    echo "  ✅ Created final_submission/${TEAM_NAME}_${LANG}.zip"
done


echo "============================================================"
echo "  🎉 SUBMISSION READY in ./final_submission/"
echo "============================================================"
