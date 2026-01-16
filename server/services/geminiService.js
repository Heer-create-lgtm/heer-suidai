import axios from 'axios';

let currentKeyIndex = 0;

/**
 * Get Google Gemini API keys array (read at request time after dotenv has loaded)
 */
function getApiKeys() {
    return [
        process.env.GEMINI_API_KEY_1,
        process.env.GEMINI_API_KEY_2,
        process.env.GEMINI_API_KEY_3,
        process.env.GEMINI_API_KEY_4,
        process.env.GEMINI_API_KEY_5
    ].filter(Boolean);
}

/**
 * Get next API key (rotates on rate limit)
 */
function getNextKey() {
    const keys = getApiKeys();
    const key = keys[currentKeyIndex % keys.length];
    currentKeyIndex = (currentKeyIndex + 1) % keys.length;
    return key;
}

/**
 * Generate AI recommendation using Google Gemini API (gemini-2.0-flash)
 * @param {string} prompt - The prompt for Gemini
 * @param {object} data - Data context for the recommendation
 */
export async function generateRecommendation(prompt, data = null) {
    const apiKey = getNextKey();
    const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${apiKey}`;

    const fullPrompt = data
        ? `${prompt}\n\nData Context:\n${JSON.stringify(data, null, 2)}`
        : prompt;

    try {
        const response = await axios.post(url, {
            contents: [{
                parts: [{ text: fullPrompt }]
            }]
        }, {
            headers: { 'Content-Type': 'application/json' }
        });

        const content = response.data.candidates?.[0]?.content?.parts?.[0]?.text;
        return {
            success: true,
            recommendation: content || 'No recommendation generated'
        };
    } catch (error) {
        console.error('Gemini API Error:', error.response?.data || error.message);

        // Try next key if rate limited (429)
        if (error.response?.status === 429 && getApiKeys().length > 1) {
            console.log('Rate limited, trying next key...');
            return generateRecommendation(prompt, data);
        }

        return {
            success: false,
            error: error.response?.status === 429
                ? 'Rate limited on all API keys. Please wait a minute and try again.'
                : error.message,
            recommendation: null
        };
    }
}

/**
 * Generate analysis and recommendations for Aadhaar data
 */
export async function analyzeAndRecommend(analysisType, data) {
    const prompts = {
        dashboard: `You are an AI analyst for UIDAI Aadhaar data. Analyze the following enrollment and update statistics and provide 3-5 actionable recommendations for improving coverage and efficiency. Be specific and practical.`,

        hotspots: `You are an AI analyst for UIDAI. Analyze the following low-enrollment hotspot districts and provide specific recommendations for improving Aadhaar coverage in these areas. Consider factors like infrastructure, accessibility, and awareness.`,

        anomalies: `You are an AI fraud detection analyst for UIDAI. Analyze the following data patterns and identify potential anomalies or suspicious activities. Provide risk assessment and recommended actions.`,

        forecast: `You are an AI analyst for UIDAI. Based on the historical enrollment trends provided, predict future enrollment patterns and provide recommendations for resource allocation and planning.`
    };

    const prompt = prompts[analysisType] || prompts.dashboard;
    return generateRecommendation(prompt, data);
}

export default {
    generateRecommendation,
    analyzeAndRecommend
};
