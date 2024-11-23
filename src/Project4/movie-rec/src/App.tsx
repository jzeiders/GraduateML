import { loadPyodide } from "pyodide"
import './App.css'


import { useState } from 'react'
import { Button } from "./components/ui/button"
import { Card, CardHeader, CardTitle, CardContent } from "./components/ui/card"
import { Input } from "./components/ui/input"
import { Label } from "./components/ui/label"
function App() {
  const [count, setCount] = useState(0)
  const increment = async () => {
    const pyodideInstance = await loadPyodide();
    const result = pyodideInstance.runPython(`${count} + 1`);
    setCount(result);
  }

  // Simulated myIBCF function
  const myIBCF = (ratings: { [key: string]: number }): string[] => {
    // This is a placeholder function that always returns the same 10 movies
    return [
      "The Shawshank Redemption",
      "The Godfather",
      "The Dark Knight",
      "12 Angry Men",
      "Schindler's List",
      "The Lord of the Rings: The Return of the King",
      "Pulp Fiction",
      "The Good, the Bad and the Ugly",
      "Fight Club",
      "Forrest Gump"
    ]
  }

  const sampleMovies = [
    "Inception",
    "The Matrix",
    "Interstellar",
    "Gladiator",
    "The Silence of the Lambs",
    "Saving Private Ryan",
    "The Green Mile",
    "Spirited Away",
    "Parasite",
    "Whiplash"
  ]
  const [ratings, setRatings] = useState<{ [key: string]: number }>({})
  const [recommendations, setRecommendations] = useState<string[]>([])

  const handleRatingChange = (movie: string, rating: number) => {
    setRatings(prev => ({ ...prev, [movie]: rating }))
  }

  const handleSubmit = () => {
    const recommendedMovies = myIBCF(ratings)
    setRecommendations(recommendedMovies)
  }

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
