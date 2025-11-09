import axios from 'axios'
const baseUrl = '/api/spotify'

// write spotify api requests to our back end here 

const getTest = async () => {
    const request = await axios.get(baseUrl)
    return request.data
}

export default { getTest }