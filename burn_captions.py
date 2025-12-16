import subprocess#is used to run shell commands 
import os

#using ffmpeg to embed the subtitles directly on to the video
def burn_subtitles(video_path, srt_relative_path, output_path, ffmpeg_path):
    #getting absolute paths for video and output
    video_full=os.path.abspath(video_path)
    output_full=os.path.abspath(output_path)
    

    #using the relative path for subtitles exacyly as it working in my manual command
     # → Path to ffmpeg.exe
    #f'-i "{video_full} → Input video file
    #f'-vf "subtitles={srt_relative_path}"→ Add subtitles filter + SRT path
    #f'"{output_full}"'→ Output video file with subs burned in
    command = f' "{ffmpeg_path}" -i "{video_full}" -vf "subtitles={srt_relative_path}" "{output_full}"'

    print("Running command: ")
    print(command)

    try:
        subprocess.run(command,shell=True, check=True)
        print(f"[✅] Video with burned_in captions generated: {output_full}")
    except subprocess.CalledProcessError as e:
        print(f"[i] Error burning subtitles: {e}")     

if __name__ == "__main__": 
    ffmpeg_path= r"C:\ffmpeg\bin\ffmpeg.exe"
    video_file="videos/sample.mp4"
    srt_relative_path ="captions/output.srt"

    output_video ="videos/output_video.mp4" 

    burn_subtitles(video_file, srt_relative_path, output_video, ffmpeg_path)


