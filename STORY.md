## Inspiration

We have all been there. The racing heart, blank minds, and the sudden inability to explain how two-pointers work. Tech interviews are very stressful and anxiety-inducing. I built **SmartTalk** to turn that anxiety into success. I wanted to move away from the basic Leetcode-style problem-solution approach and create a more robust feedback system using Gemini-3.

## What it does

SmartTalk is an AI-powered simulator for tech interviews.

- **Infinite Scenarios:** Using Gemini-3, it generates unique problem sets and tests your ability to write algorithms and good code. Users can generate an infinite number of coding quizzes to test their abilities.
- **Granular Feedback:** After solving each question, SmartTalk provides feedback based on your code's efficiency, style, and what an interviewer might take away from your submission.

## How we built it

- **Brain:** I integrated Gemini-3 Pro for its deep reasoning abilities, enabling users to get insightful feedback on their coding submissions.
- **Stack:** I used FastAPI, JavaScript, and React.js to create the dynamic frontend and a quick backend.
- **Memory:** I used a problem pool to speed up problem generation.

## Challenges we ran into

- The biggest challenge I ran into was **formatting**. Gemini would sometimes return the wrong format, resulting in the code failing to render on the website.
- Gemini also ran slowly sometimes, resulting in long wait times when getting feedback.

## Accomplishments that we're proud of

- Quick feedback that helps users gain deeper insights
- Creating the whole system on my own

## What we learned

I learned that LLMs can be quite faulty even with strict instructions. It showed me that I might need to use fine-tuning to get better results.

## What's next for SmartTalk

I want to add timer features as well as the ability to interact with a "digital interviewer" so users can also be emotionally ready for interviews.
