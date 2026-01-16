import express from 'express';
import { getDemographicData } from '../services/dataGovApi.js';

const router = express.Router();

// GET /api/demographic - Get demographic update data
router.get('/', async (req, res) => {
    try {
        const { limit = 100, offset = 0, state, district } = req.query;

        const filters = {};
        if (state) filters['filters[state]'] = state;
        if (district) filters['filters[district]'] = district;

        const data = await getDemographicData({ limit: parseInt(limit), offset: parseInt(offset), filters });
        res.json(data);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

export default router;
