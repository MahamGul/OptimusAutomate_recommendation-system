import ollama


def get_llm_explanation(selected_movie: str, results: list, model_type: str) -> str:
    """
    Use Ollama (llama3) to explain why movies were recommended.
    """
    try:
        formatted = "\n".join(
            [f"- {r['title']} (similarity: {round(r['score'], 3)})" for r in results]
        )

        prompt = f"""You are Binge Bot, a movie expert. The user liked "{selected_movie}".
Top recommendations: {formatted}
Write 2-3 punchy sentences explaining why these fit. Mention genre/mood/themes. End with a hype line. No bullet points."""

        response = ollama.chat(
            model="phi3:mini",
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": 180, "temperature": 0.8}
        )

        return response["message"]["content"]

    except Exception as e:
        return f"⚠️ AI explanation unavailable: {str(e)}"


def get_movie_vibe(selected_movie: str, genres: str) -> str:
    """
    Generate a short 'vibe check' for the selected movie.
    """
    try:
        prompt = f"""In exactly 1 sentence (max 20 words), describe the vibe of "{selected_movie}" (genres: {genres}).
Be creative and punchy. No quotes around the sentence."""

        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": 40, "temperature": 0.9}
        )

        return response["message"]["content"].strip()

    except:
        return ""