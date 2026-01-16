import express from 'express';
import { generateRecommendation, analyzeAndRecommend } from '../services/geminiService.js';

const router = express.Router();

// POST /api/ai/recommend - Get AI recommendation
router.post('/recommend', async (req, res) => {
    try {
        const { prompt, data } = req.body;

        if (!prompt) {
            return res.status(400).json({ error: 'Prompt is required' });
        }

        const result = await generateRecommendation(prompt, data);
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// POST /api/ai/analyze - Get AI analysis for specific feature
router.post('/analyze', async (req, res) => {
    try {
        const { type, data } = req.body;

        if (!type || !data) {
            return res.status(400).json({ error: 'Type and data are required' });
        }

        const result = await analyzeAndRecommend(type, data);
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

export default router;
