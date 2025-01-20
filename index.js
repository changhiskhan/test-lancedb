'use strict'

// Basic CRUD example
async function basicExample(uri, apiKey) {

    const lancedb = require('@lancedb/lancedb')
    // Import transformers and the all-MiniLM-L6-v2 model (https://huggingface.co/Xenova/all-MiniLM-L6-v2)
    const { pipeline } = await import('@xenova/transformers')
    const pipe = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    
    // Create embedding function from pipeline which returns a list of vectors from batch
    // sourceColumn is the name of the column in the data to be embedded
    //
    // Output of pipe is a Tensor { data: Float32Array(384) }, so filter for the vector
    const embed_fun = {}
    embed_fun.sourceColumn = 'text'
    embed_fun.embed = async function (batch) {
        let result = []
        for (let text of batch) {
            const res = await pipe(text, { pooling: 'mean', normalize: true })
            result.push(Array.from(res['data']))
        }
        return (result)
    }

    // Connects to LanceDB
    const db = await lancedb.connect({
        uri: uri,
        apiKey: apiKey, 
        region: "us-east-1"
    });

    // Test data
    const data = [
        { id: 1, text: 'Cherry', type: 'fruit' },
        { id: 2, text: 'Carrot', type: 'vegetable' },
        { id: 3, text: 'Potato', type: 'vegetable' },
        { id: 4, text: 'Apple', type: 'fruit' },
        { id: 5, text: 'Banana', type: 'fruit' }
    ]

    // Generate embeddings
    for (let row of data) {
        row.vector = (await embed_fun.embed([row.text]))[0]
    }

    // Create the table
    const table = await db.createTable("food_table", data, {mode:"overwrite"})

    // Query the table
    let query = (await embed_fun.embed(['a sweet fruit to eat']))[0]
    const results = await table
        .search(query)
        .select(["id", "text", "type"]) // omit if you want the vectors too
        .limit(2)
        .toArray()
    console.log(results)
}

const API_KEY = process.env.LANCEDB_API_KEY
const URI = process.env.LANCEDB_URI
basicExample(URI, API_KEY).then(_ => { console.log("Done!") })
