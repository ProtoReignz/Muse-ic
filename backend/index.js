require ('dotenv').config()
const express = require('express')
const SpotifyWebAPI = require('spotify-web-api-node')
const app = express()

// connection to spotifies web api
const spotifyApi = new SpotifyWebAPI({
    clientId: process.env.SPOTIFY_API_CLIENT_ID,
    clientSecret: process.env.SPOTIFY_API_CLIENT_SECRET,
    redirectUri: 'http://localhost:5173',   
})

// write api routes here 

app.get('/api/spotify', (request, response) => {
    response.send('AMONGUS')
})

const PORT = process.env.PORT
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`)
})