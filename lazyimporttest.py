import sys

print("--- EXPLICIT DEEPER DEBUGGER ---")
try:
    print("Testing base auto module...")
    from transformers.models.auto.processing_auto import AutoProcessor
    from transformers.models.auto.modeling_auto import AutoModelForMultimodalLM
    print("Success on auto mapping!")
except Exception as e:
    import traceback
    print("\n[CRITICAL FAILURE REVEALED]:")
    traceback.print_exc()
