import axios from 'axios';

const BASE_URL = 'https://api.data.gov.in/resource';

// Resource IDs from data.gov.in for Aadhaar data
const RESOURCES = {
    ENROLMENT: 'ecd49b12-3084-4521-8f7e-ca8bf72069ba',
    DEMOGRAPHIC: '19eac040-0b94-49fa-b239-4f2fd8677d53',
    BIOMETRIC: '65454dab-1517-40a3-ac1d-47d4dfe6891c'
};

/**
 * Fetch data from data.gov.in API
 * @param {string} resourceId - The resource ID to fetch
 * @param {object} options - Query options (limit, offset, filters)
 */
async function fetchData(resourceId, options = {}) {
    const API_KEY = process.env.DATA_GOV_API_KEY;

    if (!API_KEY) {
        console.warn('Warning: DATA_GOV_API_KEY not set. Using mock data.');
        return getMockData(resourceId, options);
    }

    const { limit = 100, offset = 0, filters = {} } = options;

    const params = new URLSearchParams({
        'api-key': API_KEY,
        format: 'json',
        limit: limit.toString(),
        offset: offset.toString(),
        ...filters
    });

    const url = `${BASE_URL}/${resourceId}?${params}`;

    try {
        const response = await axios.get(url);
        return {
            success: true,
            total: response.data.total,
            count: response.data.count,
            records: response.data.records || []
        };
    } catch (error) {
        console.error(`API Error for ${resourceId}:`, error.message);
        // Return mock data as fallback
        return getMockData(resourceId, options);
    }
}

/**
 * Generate mock data when API is unavailable
 */
function getMockData(resourceId, options = {}) {
    const states = [
        'Maharashtra', 'Uttar Pradesh', 'Bihar', 'West Bengal', 'Madhya Pradesh',
        'Tamil Nadu', 'Rajasthan', 'Karnataka', 'Gujarat', 'Andhra Pradesh',
        'Odisha', 'Kerala', 'Jharkhand', 'Assam', 'Punjab', 'Chhattisgarh',
        'Haryana', 'Delhi', 'Jammu and Kashmir', 'Uttarakhand', 'Himachal Pradesh',
        'Tripura', 'Meghalaya', 'Manipur', 'Nagaland', 'Goa', 'Arunachal Pradesh',
        'Mizoram', 'Sikkim', 'Telangana'
    ];

    const districts = {
        'Maharashtra': ['Mumbai', 'Pune', 'Nagpur', 'Thane', 'Nashik'],
        'Uttar Pradesh': ['Lucknow', 'Kanpur', 'Varanasi', 'Agra', 'Meerut'],
        'Bihar': ['Patna', 'Gaya', 'Bhagalpur', 'Muzaffarpur', 'Kishanganj'],
        'West Bengal': ['Kolkata', 'Howrah', 'Darjeeling', 'Siliguri', 'Asansol'],
        'Tamil Nadu': ['Chennai', 'Coimbatore', 'Madurai', 'Salem', 'Tiruchirappalli'],
        'Karnataka': ['Bengaluru', 'Mysuru', 'Mangaluru', 'Hubballi', 'Belagavi'],
        'Arunachal Pradesh': ['Itanagar', 'Tawang', 'Ziro', 'Bomdila', 'Pasighat']
    };

    const { limit = 100 } = options;
    const records = [];

    // Generate realistic mock enrollment data
    const months = ['2024-01', '2024-02', '2024-03', '2024-04', '2024-05', '2024-06',
        '2024-07', '2024-08', '2024-09', '2024-10', '2024-11', '2024-12'];

    for (let i = 0; i < Math.min(limit, states.length * 12); i++) {
        const stateIndex = i % states.length;
        const state = states[stateIndex];
        const stateDistricts = districts[state] || ['District 1', 'District 2', 'District 3'];
        const district = stateDistricts[Math.floor(Math.random() * stateDistricts.length)];
        const month = months[Math.floor(i / states.length) % 12];

        // Simulate different enrollment levels by state
        const baseEnrollment = 10000 + Math.floor(Math.random() * 50000);
        const coverageFactor = state === 'Arunachal Pradesh' ? 0.3 :
            state === 'Bihar' ? 0.5 :
                state === 'Maharashtra' ? 0.9 : 0.7;

        records.push({
            date: month,
            state: state,
            district: district,
            pincode: `${400000 + Math.floor(Math.random() * 99999)}`,
            age_0_5: Math.floor(baseEnrollment * 0.1 * coverageFactor),
            age_5_17: Math.floor(baseEnrollment * 0.25 * coverageFactor),
            age_18_greater: Math.floor(baseEnrollment * 0.65 * coverageFactor),
            // For demographic updates
            demo_age_5_17: Math.floor(baseEnrollment * 0.05 * coverageFactor),
            demo_age_17_: Math.floor(baseEnrollment * 0.15 * coverageFactor),
            // For biometric updates
            bio_age_5_17: Math.floor(baseEnrollment * 0.02 * coverageFactor),
            bio_age_17_: Math.floor(baseEnrollment * 0.08 * coverageFactor)
        });
    }

    return {
        success: true,
        total: records.length,
        count: records.length,
        records: records,
        isMockData: true
    };
}

/**
 * Get Aadhaar Monthly Enrolment Data
 * Fields: date, state, district, pincode, age_0_5, age_5_17, age_18_greater
 */
export async function getEnrolmentData(options = {}) {
    return fetchData(RESOURCES.ENROLMENT, options);
}

/**
 * Get Aadhaar Demographic Monthly Update Data
 * Fields: date, state, district, pincode, demo_age_5_17, demo_age_17_
 */
export async function getDemographicData(options = {}) {
    return fetchData(RESOURCES.DEMOGRAPHIC, options);
}

/**
 * Get Aadhaar Biometric Monthly Update Data
 * Fields: date, state, district, pincode, bio_age_5_17, bio_age_17_
 */
export async function getBiometricData(options = {}) {
    return fetchData(RESOURCES.BIOMETRIC, options);
}

/**
 * Get all data sources combined
 */
export async function getAllData(options = {}) {
    const [enrolment, demographic, biometric] = await Promise.all([
        getEnrolmentData(options),
        getDemographicData(options),
        getBiometricData(options)
    ]);

    return { enrolment, demographic, biometric };
}

export default {
    getEnrolmentData,
    getDemographicData,
    getBiometricData,
    getAllData,
    RESOURCES
};
