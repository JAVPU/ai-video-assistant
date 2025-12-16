import whisper
import srt#library for creating subtitles in the srt format and for burning the subtiteles to the video back
from datetime import timedelta

def transcribe(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    #the result havr transcribe as well as segmented time stamps for each sentence
    return result


def convert_to_srt(transcription_result):
    #extracting individual sehments from the whisper result 
    #where each segment have the start and end time and text that needs to displayed on the video
    segments = transcription_result['segments']
    subtitles = []
    for i,seg in enumerate(segments):
        subtitle = srt.Subtitle(
            index = i+1,
            start = timedelta(seconds=seg['start']),
            end = timedelta(seconds=seg['end']),
            content = seg['text'].strip()
        )

        subtitles.append(subtitle)

    return srt.compose(subtitles)

if __name__ == "__main__": 
    audio_file ="audio\sample_audio.wav"
    result = transcribe(audio_file)

    srt_content = convert_to_srt(result)

    with open("captions/output.srt","w",encoding ="utf-8") as f:
        f.write(srt_content)

    print("SRT File generated: captions/output.srt")    
  

