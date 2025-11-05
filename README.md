# Sign-Language-Detection-System

This is a beginner-friendly machine learning project that recognizes hand gestures in real-time using your webcam. It’s great for learning about computer vision and machine learning.

## What This Project Does

This system can:
- Capture hand gestures from your webcam
- Learn to recognize different sign language gestures
- Detect gestures in real-time with confidence scores

## What You Need

### Software Requirements
- Python 3.8 or higher
- A webcam
- The following Python libraries:
  - opencv-python
  - mediapipe
  - numpy
  - scikit-learn

### Installing Python Libraries

Open your terminal or command prompt and run:
```bash
pip install opencv-python mediapipe numpy scikit-learn
```

Wait for the installation to finish. You’re now ready to start!

## Understanding the Project Files

Your project has three main Python scripts:

1. **collectdatamp.py** - Collects training data by capturing your hand gestures.
2. **trainmodelmp.py** - Trains the AI model to recognize gestures.
3. **detectrealtimemp.py** - Uses the trained model to detect gestures in real-time.

## Step-by-Step Guide

### STEP 1: Collect Data for Your Gestures

Here’s where you teach the computer what each gesture looks like.

**Run the script:**
```bash
python collectdatamp.py
```

**What happens:**
1. A window opens showing your webcam feed.
2. You’ll see a green box on the screen.
3. The program will ask you to enter a gesture name.

**What to do:**
1. Type a gesture name when prompted (examples: hello, thanks, peace, yes, no).
2. Position your hand inside the green box.
3. Make the gesture clearly.
4. Press **SPACEBAR** to capture the gesture.
5. Keep capturing the same gesture from different angles and positions.
6. Aim for **50-100 captures** per gesture for best results.
7. Press **Q** when you're done with this gesture.

**Important tips:**
- Keep your entire hand visible in the frame.
- Make sure there’s good lighting.
- Use a simple background (not too cluttered).
- Hold the gesture steady when pressing SPACEBAR.
- Be consistent with how you make the gesture.

**Repeat for each gesture:**
Run the script again for each new gesture you want to teach the system. For example:
- Run once for "hello" (capture 50-100 times).
- Run again for "thanks" (capture 50-100 times).
- Run again for "peace" (capture 50-100 times).

**What happens behind the scenes:**
The script uses MediaPipe (Google's hand tracking technology) to detect 21 points on your hand. It saves the 3D positions (x, y, z coordinates) of these points for each capture. This data is stored in the `dataset` folder.

### STEP 2: Train the Model

Now you’ll teach the computer to recognize patterns in the data you collected.

**Run the script:**
```bash
python trainmodelmp.py
```

**What happens:**
1. The script reads all the gesture data you collected.
2. It trains a machine learning model (Random Forest classifier).
3. It shows you the accuracy of the model.
4. It saves the trained model to files.

**What you'll see:**
```
Loaded X samples from Y gestures.
Gestures: hello, thanks, peace
Accuracy: 95.50%
Classification Report: [detailed statistics]
Model, scaler, and gestures saved successfully!
```

**Understanding the output:**
- **Accuracy**: Higher is better (aim for 90%+).
- **Classification Report**: Shows how well each gesture was learned.
- If accuracy is low (<80%), collect more data or make gestures clearer.

**What happens behind the scenes:**
The script normalizes your captured hand positions, splits them into training and testing sets, trains a Random Forest model, and saves three files:
- `sign_language_model.pkl` - The trained AI model.
- `scaler.pkl` - Used to normalize new data.
- `gestures.pkl` - List of gesture names.

### STEP 3: Test Real-Time Detection

Now you can use your trained model to recognize gestures live!

**Run the script:**
```bash
python detectrealtimemp.py
```

**What happens:**
1. Your webcam opens.
2. The system detects your hand automatically.
3. It shows the predicted gesture name and confidence percentage.
4. Updates happen in real-time as you change gestures.

**What to do:**
1. Make one of the gestures you trained.
2. Watch the prediction appear at the top of the screen.
3. Try different gestures and see the predictions change.
4. Press **Q** to quit.

**Reading the output:**
- Text shows: "HELLO (95.3%)", meaning the system is 95.3% confident it’s the "hello" gesture.
- Higher confidence means better recognition.
- "No hand detected" appears when your hand isn't visible.

**Troubleshooting:**
- **Wrong predictions?** Make sure you’re making the gesture exactly as you did during training.
- **Low confidence?** Collect more training data for that gesture.
- **Not detecting?** Ensure your hand is well-lit and clearly visible.

## How the System Works (Simple Explanation)

Think of teaching a child to recognize hand shapes:

1. **Collection Phase**: You show the child many examples of the "hello" gesture from different angles.
2. **Learning Phase**: The child's brain learns patterns. "When fingers are arranged this way, it means hello."
3. **Recognition Phase**: When you make the gesture, the child recognizes it based on learned patterns.

Your system works in a similar way:
1. **MediaPipe** finds 21 key points on your hand (like fingertips, knuckles, wrist).
2. **The model** learns the typical arrangement of these points for each gesture.
3. **During detection**, it compares new hand positions to what it learned and makes a prediction.

## Project Folder Structure

After running all scripts, your project will look like this:

```
your-project-folder/
│
├── collectdatamp.py           (Script to collect data)
├── trainmodelmp.py             (Script to train model)
├── detectrealtimemp.py         (Script to detect gestures)
│
├── dataset/                    (Created automatically)
│   ├── hello/                  (Gesture 1 data)
│   │   ├── hello_0.csv
│   │   ├── hello_1.csv
│   │   └── ...
│   ├── thanks/                 (Gesture 2 data)
│   └── peace/                  (Gesture 3 data)
│
├── sign_language_model.pkl     (Trained model - created after training)
├── scaler.pkl                  (Data normalizer - created after training)
└── gestures.pkl                (Gesture list - created after training)
```

## Common Questions

**Q: How many gestures can I add?**  
A: As many as you want! Just make sure each is distinct enough to tell apart.

**Q: Can I add more data later?**  
A: Yes! Just run `collectdatamp.py` again for any gesture (new or existing), then retrain with `trainmodelmp.py`.

**Q: The model isn't accurate. What should I do?**  
A: 
- Collect more samples (100+ per gesture).
- Make sure gestures are visually different.
- Use consistent lighting.
- Keep your hand in the same general position.
- Retrain the model after collecting more data.

**Q: Can I delete a gesture?**  
A: Yes! Delete the gesture folder from `dataset/` and retrain the model.

**Q: Do I need to retrain every time I add data?**  
A: Yes, run `trainmodelmp.py` again whenever you add or modify training data.

## Tips for Best Results

1. **Lighting**: Work in a well-lit room. Natural daylight works best.
2. **Background**: Stand in front of a plain wall or simple background.
3. **Consistency**: Make each gesture the same way every time.
4. **Hand Position**: Keep your entire hand visible; don't let fingers go out of frame.
5. **Distance**: Stay about 2-3 feet from the camera.
6. **Data Variety**: Capture each gesture from slightly different angles and positions.
7. **Distinct Gestures**: Make sure your gestures look different from each other.

## Troubleshooting Common Issues

**Problem: "Cannot access camera."**  
- Solution: Make sure no other program is using your webcam. Close Zoom, Skype, etc.

**Problem: "No data found."**  
- Solution: Run `collectdatamp.py` first to collect training data.

**Problem: Low accuracy (below 80%).**  
- Solution: Collect more samples (aim for 100+ per gesture) and make gestures more distinct.

**Problem: Script won't run.**  
- Solution: Make sure all libraries are installed. Run the pip install command again.

**Problem: Hand not detected.**  
- Solution: Improve lighting, move closer to the camera, or adjust your hand position.

## Next Steps After Success

Once you have a working system:
1. Add more gestures to expand your vocabulary.
2. Try creating gestures for the alphabet.
3. Experiment with two-handed gestures.
4. Share your project with friends and have them try it.

## Getting Help

If you're stuck:
1. Read the error message carefully; it often tells you what's wrong.
2. Make sure you followed each step in order.
3. Check that all files are in the correct locations.
4. Verify all libraries are installed correctly.

Happy learning and experimenting with sign language recognition!
