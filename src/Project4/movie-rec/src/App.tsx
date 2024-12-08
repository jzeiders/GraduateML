'use client'

import { loadPyodide } from "pyodide"
import './App.css'
import { useEffect, useMemo, useRef, useState } from 'react'
import { Button } from "./components/ui/button"
import { Card, CardHeader, CardTitle, CardContent } from "./components/ui/card"
import { Input } from "./components/ui/input"
import { Label } from "./components/ui/label"
import { Skeleton } from "./components/ui/skeleton"
import { useToast } from "./hooks/use-toast"
import { Loader2 } from 'lucide-react'
import main from "./assets/python/main.py?raw"
import similarityMatrix from "./assets/python/similarity_matrix.csv?raw"
import movies from "./assets/python/movies.dat?raw"
const MOVIE_OPTIONS = movies.split("\n").map(line => line.split("::")).map(([id, title]) => ({
  id: `m${id}`,
  title
}))

function App() {
  const { toast } = useToast()
  const myIBCF = useRef<any>(null)
  const [isPyodideReady, setIsPyodideReady] = useState(false)
  const [isGenerating, setIsGenerating] = useState(false)
  const [ratings, setRatings] = useState<Map<string, number>>(new Map())
  const [recommendedIds, setRecommendedIds] = useState<string[]>([])
  
  const setupMyIBCF = async () => {
    try {
      const pyodideInstance = await loadPyodide({
        indexURL: "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/",
      });
      await pyodideInstance.loadPackage("pandas")
      await pyodideInstance.loadPackage("numpy")
      await pyodideInstance.runPython(main);
      myIBCF.current = await pyodideInstance.globals.get("getRunIBCF")
      setIsPyodideReady(true)
      toast({
        title: "Ready to generate recommendations",
        description: "The recommendation engine has been initialized successfully.",
      })
    } catch (error) {
      toast({
        variant: "destructive",
        title: "Failed to initialize",
        description: "There was an error setting up the recommendation engine. Please refresh the page.",
      })
    }
  }

  useEffect(() => {
    setupMyIBCF()
  }, [])

  const recommendedMovies = useMemo(() => {
    return recommendedIds.map(id => MOVIE_OPTIONS.find(movie => movie.id === id))
  }, [recommendedIds])

  const handleRatingChange = (movie: string, rating: number) => {
    if (rating >= 1 && rating <= 5) {
      setRatings(prev => new Map(prev).set(movie, rating))
    }
  }

  const handleSubmit = async () => {
    if (ratings.size === 0) {
      toast({
        variant: "destructive",
        title: "No ratings provided",
        description: "Please rate at least one movie before generating recommendations.",
      })
      return
    }

    setIsGenerating(true)
    try {
      const recommendedMovies = myIBCF.current(similarityMatrix, ratings )
      setRecommendedIds(recommendedMovies)
      toast({
        title: "Recommendations generated",
        description: "Here are your personalized movie recommendations!",
      })
    } catch (error) {
      toast({
        variant: "destructive",
        title: "Error generating recommendations",
        description: "There was an error generating your recommendations. Please try again.",
      })
      console.error(error)
    } finally {
      setIsGenerating(false)
    }
  }

  const sampleMovies = MOVIE_OPTIONS.slice(0, 100);

  return (
    <div className="min-h-screen">
      <div className="container mx-auto p-4 py-8">
        <h1 className="text-4xl font-bold mb-2 text-center">Movie Recommender</h1>
        <p className="text-muted-foreground text-center mb-8">Rate your favorite movies and get personalized recommendations</p>
        
        <div className="mb-12">
          <h2 className="text-2xl font-semibold mb-4">Rate These Movies</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {sampleMovies.map(movie => (
              <Card key={movie.id} className="w-full transition-shadow hover:shadow-lg">
                <CardHeader>
                  <CardTitle className="text-lg line-clamp-1" title={movie.title}>
                    {movie.title}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="aspect-[2/3] mb-4 bg-gray-100 rounded-md overflow-hidden">
                    <img 
                      src={`https://liangfgithub.github.io/MovieImages/${movie.id.slice(1)}.jpg?raw=true`} 
                      alt={movie.title} 
                      className="w-full h-full object-cover hover:scale-105 transition-transform" 
                    />
                  </div>
                  <div className="flex items-center gap-4">
                    <Label htmlFor={movie.id} className="whitespace-nowrap">Rating:</Label>
                    <Input
                      id={movie.id}
                      type="number"
                      min="1"
                      max="5"
                      value={ratings.get(movie.id) || ''}
                      onChange={(e) => handleRatingChange(movie.id, Number(e.target.value))}
                      className="w-20"
                      placeholder="1-5"
                    />
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
          <div className="mt-8 text-center">
            <Button 
              onClick={handleSubmit} 
              disabled={!isPyodideReady || isGenerating}
              size="lg"
              className="min-w-[200px]"
            >
              {!isPyodideReady ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Initializing...
                </>
              ) : isGenerating ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Generating...
                </>
              ) : (
                'Get Recommendations'
              )}
            </Button>
          </div>
        </div>

        {(recommendedIds.length > 0 || isGenerating) && (
          <div>
            <h2 className="text-2xl font-semibold mb-6">Your Recommendations</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-6">
              {isGenerating ? (
                Array(5).fill(0).map((_, i) => (
                  <Card key={i} className="w-full">
                    <CardHeader>
                      <Skeleton className="h-4 w-3/4" />
                    </CardHeader>
                    <CardContent>
                      <Skeleton className="aspect-[2/3] w-full" />
                    </CardContent>
                  </Card>
                ))
              ) : (
                recommendedMovies.map(movie => (
                  <Card key={movie?.id} className="w-full transition-shadow hover:shadow-lg">
                    <CardHeader>
                      <CardTitle className="text-lg line-clamp-1" title={movie?.title}>
                        {movie?.title}
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="aspect-[2/3] bg-gray-100 rounded-md overflow-hidden">
                        <img 
                          src={`https://liangfgithub.github.io/MovieImages/${movie?.id.slice(1)}.jpg?raw=true`} 
                          alt={movie?.title} 
                          className="w-full h-full object-cover hover:scale-105 transition-transform"
                        />
                      </div>
                    </CardContent>
                  </Card>
                ))
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default App

