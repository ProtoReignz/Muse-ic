import { useState, useEffect } from 'react'
import spotifyServices from './services/spotify.js'

function App() {
    const [data, setData] = useState('')

    // use effect to get data first time
    useEffect(() => {
    const handleTestData = async () => {
        try {
            const testData = await spotifyServices.getTest()
            console.log(testData)
            setData(testData)
        } catch (e) {
            console.log(e)
        }
    } 
    handleTestData()
    
    }, [])

    // rendering front end
    // connect all components here
    return (
        <h1 className="text-3xl font-bold underline">{data}</h1>
    )
}

export default App
