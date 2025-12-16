#moviepy is a pkg which help us in extraction of audio from a video 
from moviepy.editor import VideoFileClip
#VideoFileClip for loading and manupulating the videos
import os #using for file operations like for fetching the video and saving the audio 

#the functio for extracting audio
#video_path is the path of the input video and output_audio_path is where we are going to store the audio file
def extract_audio(video_path,output_audio_path):
    try:
        video=VideoFileClip(video_path)
        audio=video.audio
        if audio:
            audio.write_audiofile(output_audio_path)
            print(f"Audio Extracted Successfully:{output_audio_path}")
        else:
            print(f"[!] No audio track found in the video")
    except Exception as e:
        print(f"[!] Error  while Extracting The Audio:{e}")

if __name__ == "__main__": 
  #======for my group members write the path and file name carefully according to what you have written===========
  video_file="videos/sample.mp4"
  audio_output= "audio/sample_audio.wav"

  #making sure that in the dir the audio output file exista if it doesnot then error
  os.makedirs(os.path.dirname(audio_output), exist_ok=True) 

  extract_audio(video_file,audio_output)

