const form = document.getElementById('hashtagForm');
const resultDiv = document.getElementById('result');

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const hashtag = document.getElementById('hashtagInput').value;

    try {
        const response = await axios.post('/predict', { hashtag });
        const probability = response.data.probability;

        resultDiv.innerHTML = `
            <div class="alert alert-success" role="alert">
                The virality probability for <strong>${hashtag}</strong> is <strong>${(probability * 100).toFixed(2)}%</strong>.
            </div>
        `;
    } catch (error) {
        resultDiv.innerHTML = `
            <div class="alert alert-danger" role="alert">
                An error occurred while predicting. Please try again later.
            </div>
        `;
    }
});