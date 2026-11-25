# Video demo script (about 7 minutes)

Record your screen with the deployed app open in one tab and the notebook in another.
Keep the camera on if your facilitator asks for it.

| # | Time | Show | Say (roughly) |
|---|------|------|---------------|
| 1 | 0:00 | README on GitHub | "This project extends my Intro to ML phishing URL detector. Instead of the URL, it looks at a **screenshot** of the page and predicts which brand the page imitates, out of 14 brands plus 'other'. The data is Phish-IRIS, 2,849 screenshots." |
| 2 | 0:40 | Notebook, section 2 (audit table) | "Before training I hashed every image: 400 were exact duplicates and 204 official test images were copies of training images. I removed them and split by near-duplicate group, so the test scores are honest." |
| 3 | 1:30 | Notebook, sections 3–4 | Walk through the preprocessing steps, then the three feature charts and their stories: class balance, brand colour, layout (whitespace/edge density). |
| 4 | 2:30 | Notebook, sections 5–7 | Model choice (MobileNetV2 transfer learning, two phases), learning curves, metrics table, confusion matrix, ROC. Name the weak classes honestly. |
| 5 | 3:30 | Deployed app → **Overview** | "This is the deployed model. The tiles show it is online, its uptime, the model version and live latency." |
| 6 | 3:50 | **Predict** tab | Upload a PayPal phishing screenshot, then a normal website screenshot. Read out the verdict, confidence and phishing score. |
| 7 | 4:30 | **Data insights** tab | Briefly show that the same three visualizations are live in the app. |
| 8 | 4:50 | **Upload & retrain** tab | Upload a .zip of labelled screenshots (folders named after classes). Show "N waiting for retraining". Press **Retrain now** and wait for the log to say promoted or rejected. |
| 9 | 5:40 | **Model performance** tab | Show the new version in the *Model versions* table and the updated metrics. "This is the evaluation of the model in production." |
| 10 | 6:00 | Terminal + `locust/results/scaling.png` | Run `docker compose up -d --scale api=4`, open Locust, start the swarm, and show the results table: latency and throughput with 1, 2 and 4 containers. |
| 11 | 6:40 | README | Wrap up: limitations (2018 data, login-page bias) and next steps. |

Tips
* Keep 3–4 screenshots ready on your desktop: one PayPal, one Microsoft, one DHL and one normal website.
* Make the retraining zip ahead of time: `amazon/` and `linkedin/` folders with about 10 images each.
* If the free cloud host is asleep, open the URL a minute before recording so it wakes up.
