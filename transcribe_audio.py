#whisper is a openAI model that helps us to convert speechto text 
import whisper

# thsi func takes the audio that was extracted and then convert the speech to text using the base model 
def transcribe(audio_path):
    model = whisper.load_model("base")
    
    #callimg for the ranscribe method a
    result = model.transcribe(audio_path)

    #printing the trnanscribe
    print("\n Transcription Result")
    print(result['text'])

    return result

if __name__ == "__main__": 
#======for my group members write the path and file name carefully according to what you have written===========
    print("Loading1...")
    audio_file = "audio\sample_audio.wav"
    print("Loading2...")
    transcribe(audio_file)

# import whisper

# audio_file = r"audio\sample_audio.wav"  # raw string

# print("Loading model...")
# model = whisper.load_model("tiny", device="cpu")
# print("Model loaded. Starting transcription...")

# result = model.transcribe(audio_file)
# print("\nTranscription result:")
# print(result['text'])
