import { loadPyodide } from "pyodide"
import './App.css'


import { useEffect, useMemo, useRef, useState } from 'react'
import { Button } from "./components/ui/button"
import { Card, CardHeader, CardTitle, CardContent } from "./components/ui/card"
import { Input } from "./components/ui/input"
import { Label } from "./components/ui/label"
import main from "./assets/python/main.py?raw"
import similarityMatrix from "./assets/python/similarity_matrix.csv?raw"
import movies from "./assets/python/movies.dat?raw"


const MOVIE_OPTIONS = movies.split("\n").map(line => line.split("::")).map(([id, title, genres]) => ({
  id: `m${id}`,
  title
}))


function App() {

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
    setRecommendedIds(recommendedMovies)
  }
  useEffect(() => {
    setupMyIBCF()
  }, [])

  const [ratings, setRatings] = useState<Map<string, number>>(new Map())
  const [recommendedIds, setRecommendedIds] = useState<string[]>([])
  
  const recommendedMovies = useMemo(() => {
    return recommendedIds.map(id => MOVIE_OPTIONS.find(movie => movie.id === id))
  }, [recommendedIds])

  const handleRatingChange = (movie: string, rating: number) => {
    setRatings(prev => new Map(prev).set(movie, rating))
  }

  const handleSubmit = () => {
    const recommendedMovies = myIBCF.current(similarityMatrix, ratings)
    setRecommendedIds(recommendedMovies)
  }
  const sampleMovies = MOVIE_OPTIONS.slice(0, 10);

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-3xl font-bold mb-6">Movie Recommender</h1>
      <div className="mb-8">
        <h2 className="text-2xl font-semibold mb-4">Rate These Movies</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {sampleMovies.map(movie => (
            <Card key={movie.id} className="w-full">
              <CardHeader>
                <CardTitle className="text-lg">{movie.title}</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="aspect-w-2 aspect-h-3 mb-4">
                  <img src={`https://liangfgithub.github.io/Proj/F24_Proj1/movies/${movie.id.slice(1)}.jpg`} alt={movie.title} className="w-full h-full object-cover" />
                </div>
               
                <div className="flex items-center">
                  <Label htmlFor={movie.id} className="mr-2">Rating:</Label>
                  <Input
                    id={movie.id}
                    type="number"
                    min="1"
                    max="5"
                    value={ratings.get(movie.id) || ''}
                    onChange={(e) => handleRatingChange(movie.id, Number(e.target.value))}
                    className="w-16"
                  />
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
        <Button onClick={handleSubmit} className="mt-6">Get Recommendations</Button>
      </div>

      {recommendedIds.length > 0 && (
        <div>
          <h2 className="text-2xl font-semibold mb-4">Your Recommendations</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4">
            {recommendedMovies.map(movie => (
              <Card key={movie?.id} className="w-full">
                <CardHeader>
                  <CardTitle className="text-lg">{movie?.title}</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="aspect-w-2 aspect-h-3 mb-4">
                    <img src={`https://liangfgithub.github.io/Proj/F24_Proj1/movies/${movie?.id.slice(1)}.jpg`} alt={movie?.title} className="w-full h-full object-cover" />
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
