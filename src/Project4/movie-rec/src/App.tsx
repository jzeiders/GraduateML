import { loadPyodide } from "pyodide"
import './App.css'


import { useEffect, useRef, useState } from 'react'
import { Button } from "./components/ui/button"
import { Card, CardHeader, CardTitle, CardContent } from "./components/ui/card"
import { Input } from "./components/ui/input"
import { Label } from "./components/ui/label"
import main from "./assets/python/main.py?raw"
import similarityMatrix from "./assets/python/similarity_matrix.csv?raw"


function App() {
  const [_count, setCount] = useState(0)
  const myIBCF = useRef<any>(null)

  const setupMyIBCF = async () => {
    const pyodideInstance = await loadPyodide({
      indexURL: "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/",
    });
    await pyodideInstance.loadPackage("pandas")
    await pyodideInstance.loadPackage("numpy")
    await pyodideInstance.runPython(main);
    myIBCF.current = await pyodideInstance.globals.get("getHypoResultTest")
    const recommendedMovies = myIBCF.current(similarityMatrix, new Map([["m1", 5], ["m2", 4], ["m3", 3]]))
    setRecommendations(recommendedMovies)
  }
  useEffect(() => {
    setupMyIBCF()
  }, [])

  const [ratings, setRatings] = useState<Map<string, number>>(new Map())
  const [recommendations, setRecommendations] = useState<string[]>([])

  const handleRatingChange = (movie: string, rating: number) => {
    setRatings(prev => new Map(prev).set(movie, rating))
  }

  const handleSubmit = () => {
    const recommendedMovies = myIBCF.current(similarityMatrix, ratings)
    setRecommendations(recommendedMovies)
  }
  const sampleMovies = [];

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-3xl font-bold mb-6">Movie Recommender</h1>
      <div className="mb-8">
        <h2 className="text-2xl font-semibold mb-4">Rate These Movies</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {sampleMovies.map(movie => (
            <Card key={movie} className="w-full">
              <CardHeader>
                <CardTitle className="text-lg">{movie}</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="aspect-w-2 aspect-h-3 mb-4">
                  <img
                    src={`/placeholder.svg?height=450&width=300&text=${encodeURIComponent(movie)}`}
                    alt={movie}
                    width={300}
                    height={450}
                    className="rounded-md object-cover"
                  />
                </div>
                <div className="flex items-center">
                  <Label htmlFor={movie} className="mr-2">Rating:</Label>
                  <Input
                    id={movie}
                    type="number"
                    min="1"
                    max="5"
                    value={ratings[movie] || ''}
                    onChange={(e) => handleRatingChange(movie, Number(e.target.value))}
                    className="w-16"
                  />
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
        <Button onClick={handleSubmit} className="mt-6">Get Recommendations</Button>
      </div>

      {recommendations.length > 0 && (
        <div>
          <h2 className="text-2xl font-semibold mb-4">Your Recommendations</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4">
            {recommendations.map(movie => (
              <Card key={movie} className="w-full">
                <CardHeader>
                  <CardTitle className="text-lg">{movie}</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="aspect-w-2 aspect-h-3">
                    <img
                      src={`/placeholder.svg?height=450&width=300&text=${encodeURIComponent(movie)}`}
                      alt={movie}
                      width={300}
                      height={450}
                      className="rounded-md object-cover"
                    />
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default App
