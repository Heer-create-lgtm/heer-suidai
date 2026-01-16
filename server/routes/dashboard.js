import express from 'express';
import { getAllData } from '../services/dataGovApi.js';

const router = express.Router();

// GET /api/dashboard - Get combined data for dashboard
router.get('/', async (req, res) => {
    try {
        const { limit = 50 } = req.query;
        const data = await getAllData({ limit: parseInt(limit) });
        res.json(data);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

export default router;
