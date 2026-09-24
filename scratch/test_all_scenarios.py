import sys
import time
from pathlib import Path
from PIL import Image

from medtrustxai.config import Config
from medtrustxai.modalities import MODALITIES, normalize_modality
from medtrustxai.app.gradio_app import analyze, chat_followup, PROMPT_PRESETS

def p(*args, **kwargs):
    print(*args, **kwargs, flush=True)

def main():
    p("=== STARTING COMPREHENSIVE SCENARIO TESTING ===")
    config_path = "config/default.yaml"
    
    # Create dummy medical images (RGB, 256x256) for modalities without downloads
    dummy_img_path = Path("scratch/dummy_test_image.png")
    dummy_img_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (256, 256), color=(200, 200, 200))
    img.save(dummy_img_path)
    
    # Real test image if available
    cxr_path = Path("data/test_cxr/images/cxr_001_normal_pa.jpg")
    real_img = Image.open(cxr_path) if cxr_path.exists() else img
    
    scenarios_tested = 0
    failures = []
    
    # Test 1: Empty Image scenario across all modalities
    p("\n--- Test 1: Empty image scenarios ---")
    for mod in MODALITIES:
        try:
            res = analyze(None, mod, "Full report", "", False, 32, config_path)
            scenarios_tested += 1
            assert len(res) == 6, f"Expected 6 outputs, got {len(res)}"
            assert res[4] == "Waiting for image…", f"Unexpected status: {res[4]}"
            p(f"  [OK] Modality: {mod} with image=None")
        except Exception as e:
            failures.append((f"Empty image - {mod}", str(e)))
            p(f"  [FAIL] Modality: {mod} empty image: {e}")

    # Test 2: Custom Question empty prompt scenario
    p("\n--- Test 2: Custom Question empty prompt scenarios ---")
    for mod in MODALITIES:
        try:
            res = analyze(img, mod, "Custom question", "", False, 32, config_path)
            scenarios_tested += 1
            assert len(res) == 6, f"Expected 6 outputs, got {len(res)}"
            assert res[4] == "Enter a custom question.", f"Unexpected status: {res[4]}"
            p(f"  [OK] Modality: {mod} empty custom prompt")
        except Exception as e:
            failures.append((f"Empty custom prompt - {mod}", str(e)))
            p(f"  [FAIL] Modality: {mod} custom prompt: {e}")

    # Test 3: Run analysis for ALL presets across ALL modalities (no XAI)
    p("\n--- Test 3: Analysis across all presets per modality ---")
    for mod in MODALITIES:
        presets = PROMPT_PRESETS.get(mod, {})
        for preset_name, default_prompt in presets.items():
            if preset_name == "Custom question":
                custom_text = "What specific findings are visible in this medical test?"
                p_arg = preset_name
            else:
                custom_text = ""
                p_arg = preset_name
            
            p(f"  Testing {mod} -> preset: '{preset_name}'...")
            t0 = time.time()
            try:
                res = analyze(real_img, mod, p_arg, custom_text, False, 16, config_path)
                scenarios_tested += 1
                assert len(res) == 6, f"Expected 6 outputs, got {len(res)}"
                chatbot, gradcam, findings, prompt_out, status, faith = res
                assert isinstance(chatbot, list), "Chatbot output must be list"
                assert status.startswith("Done in"), f"Unexpected status: {status}"
                assert len(findings) > 0, "Findings output should not be empty"
                p(f"    [OK] ({time.time()-t0:.2f}s) Findings len: {len(findings)}")
            except Exception as e:
                failures.append((f"Analyze - {mod} - {preset_name}", str(e)))
                p(f"    [FAIL] {mod} - {preset_name}: {e}")

    # Test 4: Run analysis WITH Grad-CAM (XAI) for radiology
    p("\n--- Test 4: Analysis WITH Grad-CAM (XAI) ---")
    try:
        t0 = time.time()
        res = analyze(real_img, "radiology", "Findings only", "", True, 16, config_path)
        scenarios_tested += 1
        assert len(res) == 6, f"Expected 6 outputs, got {len(res)}"
        chatbot, gradcam, findings, prompt_out, status, faith = res
        assert gradcam is not None, "Grad-CAM image should be returned"
        assert "Score" in faith, f"Faithfulness score expected in '{faith}'"
        p(f"  [OK] Radiology with Grad-CAM ({time.time()-t0:.2f}s). Faithfulness: {faith}")
    except Exception as e:
        failures.append(("Grad-CAM XAI", str(e)))
        p(f"  [FAIL] Grad-CAM XAI: {e}")

    # Test 5: Chat Followup Scenarios
    p("\n--- Test 5: Chat Followup Scenarios ---")
    history, msg, status = chat_followup(real_img, "radiology", "Findings text", "Initial prompt", [], "", 16, config_path)
    assert status == "Type a question below.", f"Unexpected chat status: {status}"
    history, msg, status = chat_followup(real_img, "radiology", "", "Initial prompt", [], "What about heart size?", 16, config_path)
    assert status == "Click Analyze first.", f"Unexpected chat status: {status}"
    history, msg, status = chat_followup(None, "radiology", "Findings text", "Initial prompt", [], "What about heart size?", 16, config_path)
    assert status == "Upload an image first.", f"Unexpected chat status: {status}"
    
    p("  Testing valid chat followup...")
    history, msg, status = chat_followup(real_img, "radiology", "No acute osseous abnormality.", "Analyze chest radiograph.", [], "Is there cardiomegaly?", 16, config_path)
    assert status.startswith("Replied in"), f"Unexpected chat reply status: {status}"
    assert len(history) == 2, f"Expected 2 chat turns, got {len(history)}"
    p(f"  [OK] Valid chat response: {history[-1]['content']}")
    scenarios_tested += 4

    p(f"\n==========================================")
    p(f"SUMMARY: Tested {scenarios_tested} scenarios.")
    if failures:
        p(f"FAILURES ({len(failures)}):")
        for name, err in failures:
            p(f"  - {name}: {err}")
        sys.exit(1)
    else:
        p("ALL SCENARIOS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
