import { createSlice, PayloadAction } from '@reduxjs/toolkit'

interface MediaItem {
  id: string
  title: string
  type: 'movie' | 'tv' | 'music' | 'book' | 'photo'
  path: string
  size: number
  added: string
  watched: boolean
  rating?: number
  poster?: string
}

interface MediaStats {
  totalMovies: number
  totalShows: number
  totalEpisodes: number
  totalMusic: number
  totalBooks: number
  totalSize: string
  recentlyAdded: number
  watchTime: string
}

interface MediaState {
  items: MediaItem[]
  stats: MediaStats | null
  loading: boolean
  error: string | null
  filter: {
    type: string | null
    search: string
    sortBy: 'added' | 'title' | 'size' | 'rating'
    sortOrder: 'asc' | 'desc'
  }
}

const initialState: MediaState = {
  items: [],
  stats: null,
  loading: false,
  error: null,
  filter: {
    type: null,
    search: '',
    sortBy: 'added',
    sortOrder: 'desc',
  },
}

const mediaSlice = createSlice({
  name: 'media',
  initialState,
  reducers: {
    setMediaItems: (state, action: PayloadAction<MediaItem[]>) => {
      state.items = action.payload
      state.loading = false
      state.error = null
    },
    setMediaStats: (state, action: PayloadAction<MediaStats>) => {
      state.stats = action.payload
    },
    updateFilter: (state, action: PayloadAction<Partial<MediaState['filter']>>) => {
      state.filter = { ...state.filter, ...action.payload }
    },
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.loading = action.payload
    },
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload
      state.loading = false
    },
  },
})

export const { setMediaItems, setMediaStats, updateFilter, setLoading, setError } = mediaSlice.actions
export default mediaSlice.reducer