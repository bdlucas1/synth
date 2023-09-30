import soundsliceapi as ss
import yaml
import sys
import os
import time

# open connection
yaml_fn = os.path.join(os.path.dirname(__file__), ".ss.yaml")
credentials = yaml.safe_load(open(yaml_fn, "r"))
client = ss.Client(credentials["app_id"], credentials["password"])

def get_scorehash(slice_name):
    slices = client.list_slices()
    slice = next(slice for slice in slices if slice["name"] == slice_name)
    scorehash = slice["scorehash"]
    print(slice_name, scorehash)
    #print(client.get_slice_recordings(scorehash))
    return scorehash

def upload_mp3(scorehash, recording_name, recording_file):
    for recording in get_recordings(scorehash, recording_name):
        response = client.delete_recording(recording["id"])
        print("deleting recording_id", recording["id"], response)
    response = client.create_recording(
        scorehash,
        ss.Constants.SOURCE_MP3_UPLOAD,
        filename=recording_file,
        name=recording_name
    )
    print("uploaded", recording_file, recording_name, response)
    while True:
        recording = get_recordings(scorehash, recording_name)[0]
        print(recording["id"], recording["name"], recording["status"])
        if recording["status"] == "ready":
            break
        time.sleep(1)

def get_recordings(scorehash, name):
    recordings = client.get_slice_recordings(scorehash)
    return [recording for recording in recordings if recording["name"] == name]

def put_syncpoints(recording_id, sync_file):
    syncpoints = open(sync_file, "r").read()
    response = client.put_recording_syncpoints(recording_id, syncpoints)
    print("uploaded", sync_file, response)

slice_name = sys.argv[1]
recording_name = sys.argv[2]

recording_file = os.path.join(os.path.dirname(__file__), recording_name)
sync_file = os.path.join(os.path.dirname(__file__), recording_name.replace(".mp3", ".sync"))

scorehash = get_scorehash(slice_name)
upload_mp3(scorehash, recording_name, recording_file)
recording = get_recordings(scorehash, recording_name)[0]
put_syncpoints(recording["id"], sync_file)
