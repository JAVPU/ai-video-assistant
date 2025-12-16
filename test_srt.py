try:
    import srt
    print("✅ srt is installed! Version:", srt.__version__)
except ImportError:
    print("❌ srt is NOT installed.")
