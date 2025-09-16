# ELAN Basics (Simple Guide)

ELAN (EUDICO Linguistic Annotator) is a tool for creating time-aligned annotations on audio/video. This project supports importing `.eaf` files and exporting transcriptions back to ELAN.

## Download and Install

- Download from the Max Planck Institute: `https://archive.mpi.nl/tla/elan/download`
- Install and launch ELAN.

## Initial UI

![Screenshot 2025-09-16 105405.png](assets/Screenshot%202025-09-16%20105405.png)

This is the initial UI when you click on an `.eaf` file. The interface shows:
- Audio waveform display with timeline
- Multiple speaker tiers (SPEAKER_01, SPEAKER_02, etc.)
- Playback controls
- Annotation table on the right

## Understanding Chunks

![Screenshot 2025-09-16 105406.png](assets/Screenshot%202025-09-16%20105406.png)

The marked rectangular parts in the annotation tiers are called **chunks**. You can:
- **Control duration**: Hold `Alt` + drag with mouse to adjust chunk boundaries
- **Change text**: Double-click on a chunk to edit the transcription text

## My Know-How Method for Easy Annotation

Here's an efficient workflow for creating annotations:

### Step 1: Adjust Chunk Duration with Segmentation Mode

![Screenshot 2025-09-16 113320.png](assets/Screenshot%202025-09-16%20113320.png)

1. Go to `Options` → `Segmentation Mode`
2. In the left panel, select "One keystroke per annotation, fixed duration"
3. Set duration (e.g., 1000ms = 1 second)
4. Click on the waveform to create segments with consistent duration
5. This creates the time boundaries for your annotations

### Step 2: Correct Annotation Text using Transcription Mode

![Screenshot 2025-09-16 113504.png](assets/Screenshot%202025-09-16%20113504.png)

1. Go to `Options` → `Transcription Mode`
2. Click on each chunk to edit the text content
3. Type or correct the Greek transcription
4. Press Enter to save and move to the next chunk

## Additional Tips

### Creating Tiers (Speakers)
1. `Tier` → `Add New Tier…`
2. Name: `SPEAKER_01`, `SPEAKER_02`, etc.
3. Annotation type: `Orthography`

### Linking Audio
1. `File` → `Edit Linked Files` → `Add`
2. Select your audio file (e.g., `data/test2.wav`)

### Saving Your Work
- `File` → `Save As` → choose name like `data/test2_transcribed.eaf`

## Using EAF with This Project

### Convert EAF to Dataset
```bash
python process_prepair_datasets.py --input data/test2_transcribed.eaf --dataset-url <user/repo> --output-dir data/test2_dataset --verbose
```

### Transcribe Audio and Export to ELAN
```bash
python process_transcribe.py --input_file data/test2.wav --model-size base --language Greek --output data/out.eaf diarized
```

## Common Conventions

- Tier names: `SPEAKER_01`, `SPEAKER_02`, etc.
- Keep annotation text in the `value` field
- Time units are in milliseconds
- Greek text is fully supported

## Troubleshooting

- **Audio not playing**: Relink media via `File` → `Edit Linked Files`
- **Missing Greek fonts**: Ensure system fonts support Greek
- **Exported EAF empty**: Check that annotations exist and were saved

## References

- ELAN manual: `https://archive.mpi.nl/tla/elan/documentation`
