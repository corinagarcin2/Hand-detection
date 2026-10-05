// ============================================================
// IMPORT MEDIAPIPE
// ============================================================

import {
    HandLandmarker,
    FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/+esm";


// ============================================================
// HTML ELEMENTS
// ============================================================

const video = document.getElementById("webcam");
const canvas = document.getElementById("canvas");

const ctx = canvas.getContext("2d");

const startButton = document.getElementById("startButton");
const clearButton = document.getElementById("clearButton");

const handText = document.getElementById("hand");
const fingersText = document.getElementById("fingers");
const gestureText = document.getElementById("gesture");
const confidenceText = document.getElementById("confidence");


// ============================================================
// VARIABLES
// ============================================================

let handLandmarker = null;

let lastVideoTime = -1;

let previousPoint = null;

let drawingLines = [];


// ============================================================
// HAND CONNECTIONS
// ============================================================

const HAND_CONNECTIONS = [

    [0, 1], [1, 2], [2, 3], [3, 4],

    [0, 5], [5, 6], [6, 7], [7, 8],

    [5, 9], [9, 10], [10, 11], [11, 12],

    [9, 13], [13, 14], [14, 15], [15, 16],

    [13, 17], [17, 18], [18, 19], [19, 20],

    [0, 17]

];


// ============================================================
// LOAD MEDIAPIPE
// ============================================================

async function createHandLandmarker() {

    gestureText.textContent = "Loading...";

    const vision = await FilesetResolver.forVisionTasks(

        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"

    );


    handLandmarker =
        await HandLandmarker.createFromOptions(

            vision,

            {

                baseOptions: {

                    modelAssetPath:

                    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

                },

                runningMode: "VIDEO",

                numHands: 2,

                minHandDetectionConfidence: 0.5,

                minHandPresenceConfidence: 0.5,

                minTrackingConfidence: 0.5

            }

        );


    gestureText.textContent = "Ready";

    console.log("MediaPipe loaded!");

}


// ============================================================
// START CAMERA
// ============================================================

async function startCamera() {

    try {

        if (!handLandmarker) {

            gestureText.textContent = "Loading AI...";

            await createHandLandmarker();

        }


        const stream =
            await navigator.mediaDevices.getUserMedia({

                video: {

                    width: 1280,

                    height: 720

                }

            });


        video.srcObject = stream;


        await video.play();


        // Make canvas match camera resolution

        canvas.width = video.videoWidth;

        canvas.height = video.videoHeight;


        startButton.textContent = "Camera Running";

        startButton.disabled = true;


        predictWebcam();

    }

    catch (error) {

        console.error(error);

        gestureText.textContent = "Error";

        alert(
            "Could not start hand detection. Check camera permission and console."
        );

    }

}


// ============================================================
// FINGER DETECTION
// ============================================================

function getFingers(landmarks, handLabel) {

    const fingers = [];


    // --------------------------------------------------------
    // THUMB
    // --------------------------------------------------------

    const thumbTip = landmarks[4];

    const thumbJoint = landmarks[3];


    let thumbUp;


    if (handLabel === "Right") {

        thumbUp =
            thumbTip.x < thumbJoint.x;

    }

    else {

        thumbUp =
            thumbTip.x > thumbJoint.x;

    }


    fingers.push(thumbUp);


    // --------------------------------------------------------
    // INDEX / MIDDLE / RING / PINKY
    // --------------------------------------------------------

    const tips = [8, 12, 16, 20];

    const joints = [6, 10, 14, 18];


    for (let i = 0; i < tips.length; i++) {

        const fingerUp =

            landmarks[tips[i]].y
            <
            landmarks[joints[i]].y;


        fingers.push(fingerUp);

    }


    return fingers;

}


// ============================================================
// GESTURE RECOGNITION
// ============================================================

function recognizeGesture(fingers, landmarks) {

    const thumb = fingers[0];

    const index = fingers[1];

    const middle = fingers[2];

    const ring = fingers[3];

    const pinky = fingers[4];


    const count =
        fingers.filter(Boolean).length;


    // FIST

    if (count === 0) {

        return "FIST";

    }


    // OPEN HAND

    if (count === 5) {

        return "OPEN HAND";

    }


    // PEACE

    if (

        index &&
        middle &&
        !ring &&
        !pinky

    ) {

        return "PEACE";

    }


    // THUMBS UP
    // Check this BEFORE pointing because both
    // gestures can have only one raised finger.

    if (

        thumb &&
        !index &&
        !middle &&
        !ring &&
        !pinky &&
        landmarks[4].y < landmarks[2].y

    ) {

        return "THUMBS UP";

    }


    // POINTING

    if (

        index &&
        !middle &&
        !ring &&
        !pinky

    ) {

        return "POINTING";

    }


    return "UNKNOWN";

}


function drawHand(landmarks) {

    // ==========================================
    // NEON PINK HAND CONNECTIONS
    // ==========================================

    ctx.strokeStyle = "#ff2d95";

    // Make the skeleton much bolder
    ctx.lineWidth = 6;

    ctx.lineCap = "round";
    ctx.lineJoin = "round";


    for (const [start, end] of HAND_CONNECTIONS) {

        const startPoint = landmarks[start];
        const endPoint = landmarks[end];

        ctx.beginPath();

        ctx.moveTo(
            startPoint.x * canvas.width,
            startPoint.y * canvas.height
        );

        ctx.lineTo(
            endPoint.x * canvas.width,
            endPoint.y * canvas.height
        );

        ctx.stroke();
    }


    // ==========================================
    // NEON PINK LANDMARK POINTS
    // ==========================================

    for (const landmark of landmarks) {

        const x = landmark.x * canvas.width;
        const y = landmark.y * canvas.height;


        // Large neon pink circle
        ctx.beginPath();

        ctx.arc(
            x,
            y,
            10,
            0,
            Math.PI * 2
        );

        ctx.fillStyle = "#ff2d95";

        ctx.fill();


        // Small light center
        ctx.beginPath();

        ctx.arc(
            x,
            y,
            3.5,
            0,
            Math.PI * 2
        );

        ctx.fillStyle = "#fff5fa";

        ctx.fill();
    }
}

// ============================================================
// SAVE DRAWING LINE
// ============================================================

function addDrawingLine(point1, point2) {

    drawingLines.push({

        start: point1,

        end: point2

    });

}


// ============================================================
// DRAW SAVED AIR DRAWING
// ============================================================

function drawSavedLines() {

    ctx.strokeStyle = "#00d9ff";

    ctx.lineWidth = 6;

    ctx.lineCap = "round";


    for (const line of drawingLines) {

        ctx.beginPath();


        ctx.moveTo(

            line.start.x,

            line.start.y

        );


        ctx.lineTo(

            line.end.x,

            line.end.y

        );


        ctx.stroke();

    }

}


// ============================================================
// MAIN DETECTION LOOP
// ============================================================

function predictWebcam() {

    if (!handLandmarker) {

        return;

    }


    // Only process a new video frame

    if (video.currentTime !== lastVideoTime) {

        lastVideoTime = video.currentTime;


        const results =
            handLandmarker.detectForVideo(

                video,

                performance.now()

            );


        // Clear canvas

        ctx.clearRect(

            0,

            0,

            canvas.width,

            canvas.height

        );


        // Redraw saved drawing

        drawSavedLines();


        // ----------------------------------------------------
        // HAND FOUND
        // ----------------------------------------------------

        if (

            results.landmarks &&
            results.landmarks.length > 0

        ) {


            const landmarks =
                results.landmarks[0];


            drawHand(landmarks);


            // ------------------------------------------------
            // HANDEDNESS
            // ------------------------------------------------

            let handLabel = "Unknown";

            let confidence = 0;


            if (

                results.handedness &&
                results.handedness.length > 0

            ) {

                const category =
                    results.handedness[0][0];


                handLabel =
                    category.categoryName;


                confidence =
                    Math.round(
                        category.score * 100
                    );

            }


            // ------------------------------------------------
            // FINGERS
            // ------------------------------------------------

            const fingers =
                getFingers(

                    landmarks,

                    handLabel

                );


            const fingerCount =
                fingers.filter(Boolean).length;


            // ------------------------------------------------
            // GESTURE
            // ------------------------------------------------

            const gesture =
                recognizeGesture(

                    fingers,

                    landmarks

                );


            // ------------------------------------------------
            // UPDATE INFORMATION PANEL
            // ------------------------------------------------

            handText.textContent =
                handLabel;


            fingersText.textContent =
                fingerCount;


            gestureText.textContent =
                gesture;


            confidenceText.textContent =
                confidence + "%";


            // ------------------------------------------------
            // AIR DRAWING
            // ------------------------------------------------

            const indexTip =
                landmarks[8];


            // Our VIDEO is mirrored with CSS.
            // Mirror X for the canvas too.

           const currentPoint = {

    x: indexTip.x * canvas.width,

    y: indexTip.y * canvas.height

};


            if (gesture === "POINTING") {


                if (previousPoint) {

                    addDrawingLine(

                        previousPoint,

                        currentPoint

                    );

                }


                previousPoint =
                    currentPoint;

            }


            else if (gesture === "FIST") {

                // Clear drawing

                drawingLines = [];

                previousPoint = null;

            }


            else {

                // OPEN HAND or other gesture
                // stops drawing

                previousPoint = null;

            }

        }


        // ----------------------------------------------------
        // NO HAND
        // ----------------------------------------------------

        else {

            handText.textContent =
                "None";


            fingersText.textContent =
                "0";


            gestureText.textContent =
                "None";


            confidenceText.textContent =
                "0%";


            previousPoint = null;

        }

    }


    // Continue detection

    requestAnimationFrame(
        predictWebcam
    );

}


// ============================================================
// START BUTTON
// ============================================================

startButton.addEventListener(

    "click",

    startCamera

);


// ============================================================
// CLEAR BUTTON
// ============================================================

clearButton.addEventListener(

    "click",

    () => {

        drawingLines = [];

        previousPoint = null;

    }

);