'use strict'

// Metadata filtering + hybrid { vector, fts } search + reranking
async function hybridSearch(uri, apiKey) {

    const lancedb = require('@lancedb/lancedb')
    // Import transformers and the all-MiniLM-L6-v2 model (https://huggingface.co/Xenova/all-MiniLM-L6-v2)
    const { pipeline } = await import('@xenova/transformers')
    const pipe = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');    
    
    const { DistanceType, Index } = await import('@lancedb/lancedb')
    
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

    // create a vector index
    await table.createIndex("vector", {
        config: lancedb.Index.ivfPq({
          distanceType: 'cosine'
        })
    });    

    await table.createIndex("text", {
        config: Index.fts(),
    });

    while ((await table.listIndices()).length < 2) {
        console.log("Indexing...")
        await new Promise(resolve => setTimeout(resolve, 1000));
    }

    // Query the table
    let query = (await embed_fun.embed(['a sweet fruit to eat']))[0]
    const result = await table
        .query()
        .nearestTo([0.1, 0.1])
        .fullTextSearch("dog")
        .where("`type`='fruit'")
        .rerank(await lancedb.rerankers.RRFReranker.create())
        .select(["text"])
        .limit(1)
        .toArray();     
    console.log(results)
}

const API_KEY = process.env.LANCEDB_API_KEY
const URI = process.env.LANCEDB_URI
hybridSearch(URI, API_KEY).then(_ => { console.log("Done!") })
