import express from 'express';
import { getEnrolmentData } from '../services/dataGovApi.js';

const router = express.Router();

// GET /api/enrolment - Get enrolment data
router.get('/', async (req, res) => {
    try {
        const { limit = 100, offset = 0, state, district } = req.query;

        const filters = {};
        if (state) filters['filters[state]'] = state;
        if (district) filters['filters[district]'] = district;

        const data = await getEnrolmentData({ limit: parseInt(limit), offset: parseInt(offset), filters });
        res.json(data);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

export default router;
