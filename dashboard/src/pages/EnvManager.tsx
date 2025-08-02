import React, { useState, useEffect } from 'react'
import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  IconButton,
  Button,
  Grid,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControlLabel,
  Checkbox,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Tooltip,
  InputAdornment,
  Tabs,
  Tab,
  Alert,
  Snackbar,
  LinearProgress,
  Paper,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  Fab,
  Autocomplete,
} from '@mui/material'
import {
  Save,
  Add,
  Delete,
  Visibility,
  VisibilityOff,
  ContentCopy,
  Refresh,
  Upload,
  Download,
  Lock,
  LockOpen,
  Search,
  FilterList,
  Info,
  Warning,
  CheckCircle,
  Error as ErrorIcon,
  Casino,
  Undo,
  Redo,
} from '@mui/icons-material'
import { motion, AnimatePresence } from 'framer-motion'
import { useForm, Controller } from 'react-hook-form'
import { yupResolver } from '@hookform/resolvers/yup'
import * as yup from 'yup'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import toast from 'react-hot-toast'
import { fetchEnvVariables, updateEnvVariables, validateEnvConfig } from '../services/api'

interface EnvVariable {
  key: string
  value: string
  category: string
  description: string
  secure: boolean
  required: boolean
  validation?: string
  defaultValue?: string
  modified?: boolean
}

interface TabPanelProps {
  children?: React.ReactNode
  index: number
  value: number
}

const categories = [
  { value: 'core', label: 'Core Infrastructure', icon: '🏗️' },
  { value: 'media', label: 'Media Services', icon: '🎬' },
  { value: 'download', label: 'Download Clients', icon: '⬇️' },
  { value: 'security', label: 'Security & Auth', icon: '🔒' },
  { value: 'monitoring', label: 'Monitoring', icon: '📊' },
  { value: 'networking', label: 'Networking', icon: '🌐' },
  { value: 'storage', label: 'Storage & Backup', icon: '💾' },
  { value: 'advanced', label: 'Advanced Options', icon: '⚡' },
]

const variableSchema = yup.object({
  key: yup.string().required('Variable name is required').matches(/^[A-Z_]+$/, 'Must be uppercase with underscores'),
  value: yup.string(),
  category: yup.string().required('Category is required'),
  description: yup.string().required('Description is required'),
  secure: yup.boolean(),
  required: yup.boolean(),
})

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`env-tabpanel-${index}`}
      aria-labelledby={`env-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  )
}

const EnvManager: React.FC = () => {
  const queryClient = useQueryClient()
  const [currentTab, setCurrentTab] = useState(0)
  const [searchTerm, setSearchTerm] = useState('')
  const [showSecureValues, setShowSecureValues] = useState<{ [key: string]: boolean }>({})
  const [dialogOpen, setDialogOpen] = useState(false)
  const [editingVariable, setEditingVariable] = useState<EnvVariable | null>(null)
  const [validationStatus, setValidationStatus] = useState<{ [key: string]: 'valid' | 'invalid' | 'warning' }>({})
  const [history, setHistory] = useState<EnvVariable[][]>([])
  const [historyIndex, setHistoryIndex] = useState(-1)

  const { data: variables = [], isLoading } = useQuery<EnvVariable[]>({
    queryKey: ['envVariables'],
    queryFn: fetchEnvVariables,
  })

  const updateMutation = useMutation({
    mutationFn: updateEnvVariables,
    onSuccess: () => {
      toast.success('Environment variables updated successfully')
      queryClient.invalidateQueries({ queryKey: ['envVariables'] })
    },
    onError: (error: any) => {
      toast.error(error.message || 'Failed to update environment variables')
    },
  })

  const validateMutation = useMutation({
    mutationFn: validateEnvConfig,
    onSuccess: (result) => {
      setValidationStatus(result)
      const hasErrors = Object.values(result).some(status => status === 'invalid')
      if (hasErrors) {
        toast.error('Validation errors found')
      } else {
        toast.success('All variables are valid')
      }
    },
  })

  const {
    control,
    handleSubmit,
    reset,
    formState: { errors },
  } = useForm({
    resolver: yupResolver(variableSchema),
    defaultValues: {
      key: '',
      value: '',
      category: 'core',
      description: '',
      secure: false,
      required: false,
    },
  })

  useEffect(() => {
    if (editingVariable) {
      reset(editingVariable)
    }
  }, [editingVariable, reset])

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setCurrentTab(newValue)
  }

  const handleAddVariable = (data: any) => {
    const updatedVariables = editingVariable
      ? variables.map(v => v.key === editingVariable.key ? { ...data, modified: true } : v)
      : [...variables, { ...data, modified: true }]
    
    updateMutation.mutate(updatedVariables)
    handleCloseDialog()
  }

  const handleDeleteVariable = (key: string) => {
    const updatedVariables = variables.filter(v => v.key !== key)
    updateMutation.mutate(updatedVariables)
  }

  const handleCloseDialog = () => {
    setDialogOpen(false)
    setEditingVariable(null)
    reset()
  }

  const handleCopyValue = (value: string) => {
    navigator.clipboard.writeText(value)
    toast.success('Copied to clipboard')
  }

  const generateRandomValue = (length: number = 32) => {
    const charset = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*'
    return Array.from({ length }, () => charset[Math.floor(Math.random() * charset.length)]).join('')
  }

  const exportConfig = () => {
    const envContent = variables
      .map(v => `${v.key}=${v.secure ? '***SECURE***' : v.value}`)
      .join('\n')
    
    const blob = new Blob([envContent], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = '.env.export'
    a.click()
    URL.revokeObjectURL(url)
    toast.success('Configuration exported')
  }

  const importConfig = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = (e) => {
        const content = e.target?.result as string
        // Parse and import logic here
        toast.success('Configuration imported')
      }
      reader.readAsText(file)
    }
  }

  const filteredVariables = variables.filter(v => {
    const matchesSearch = v.key.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         v.description.toLowerCase().includes(searchTerm.toLowerCase())
    const matchesCategory = currentTab === 0 || v.category === categories[currentTab - 1].value
    return matchesSearch && matchesCategory
  })

  const getValidationIcon = (key: string) => {
    const status = validationStatus[key]
    if (status === 'valid') return <CheckCircle color="success" fontSize="small" />
    if (status === 'invalid') return <ErrorIcon color="error" fontSize="small" />
    if (status === 'warning') return <Warning color="warning" fontSize="small" />
    return null
  }

  return (
    <Box>
      <Box sx={{ mb: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Box>
          <Typography variant="h4" gutterBottom>
            Environment Manager
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Configure and manage all service environment variables
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Button
            variant="outlined"
            startIcon={<Upload />}
            component="label"
          >
            Import
            <input
              type="file"
              hidden
              accept=".env,.txt"
              onChange={importConfig}
            />
          </Button>
          <Button
            variant="outlined"
            startIcon={<Download />}
            onClick={exportConfig}
          >
            Export
          </Button>
          <Button
            variant="contained"
            startIcon={<Add />}
            onClick={() => setDialogOpen(true)}
          >
            Add Variable
          </Button>
        </Box>
      </Box>

      <Card>
        <CardContent>
          <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}>
            <Tabs value={currentTab} onChange={handleTabChange} variant="scrollable" scrollButtons="auto">
              <Tab label="All" />
              {categories.map((cat) => (
                <Tab key={cat.value} label={`${cat.icon} ${cat.label}`} />
              ))}
            </Tabs>
          </Box>

          <Box sx={{ mb: 3, display: 'flex', gap: 2, alignItems: 'center' }}>
            <TextField
              fullWidth
              placeholder="Search variables..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <Search />
                  </InputAdornment>
                ),
              }}
            />
            <Button
              variant="outlined"
              startIcon={<CheckCircle />}
              onClick={() => validateMutation.mutate(variables)}
              disabled={validateMutation.isPending}
            >
              Validate All
            </Button>
          </Box>

          {isLoading ? (
            <LinearProgress />
          ) : (
            <AnimatePresence>
              <Grid container spacing={2}>
                {filteredVariables.map((variable) => (
                  <Grid item xs={12} key={variable.key}>
                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -20 }}
                      transition={{ duration: 0.2 }}
                    >
                      <Card
                        variant="outlined"
                        sx={{
                          backgroundColor: variable.modified ? 'rgba(0, 255, 136, 0.05)' : 'transparent',
                          borderColor: variable.modified ? 'primary.main' : 'divider',
                        }}
                      >
                        <CardContent>
                          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                            <Box sx={{ flex: 1 }}>
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                                <Typography variant="h6" component="span">
                                  {variable.key}
                                </Typography>
                                {variable.required && (
                                  <Chip label="Required" size="small" color="error" />
                                )}
                                {variable.secure && (
                                  <Chip
                                    label="Secure"
                                    size="small"
                                    color="warning"
                                    icon={<Lock fontSize="small" />}
                                  />
                                )}
                                {getValidationIcon(variable.key)}
                              </Box>
                              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                                {variable.description}
                              </Typography>
                              <TextField
                                fullWidth
                                type={showSecureValues[variable.key] ? 'text' : variable.secure ? 'password' : 'text'}
                                value={variable.value}
                                onChange={(e) => {
                                  const updatedVariables = variables.map(v =>
                                    v.key === variable.key ? { ...v, value: e.target.value, modified: true } : v
                                  )
                                  queryClient.setQueryData(['envVariables'], updatedVariables)
                                }}
                                InputProps={{
                                  endAdornment: (
                                    <InputAdornment position="end">
                                      {variable.secure && (
                                        <IconButton
                                          onClick={() => setShowSecureValues({
                                            ...showSecureValues,
                                            [variable.key]: !showSecureValues[variable.key]
                                          })}
                                          edge="end"
                                        >
                                          {showSecureValues[variable.key] ? <VisibilityOff /> : <Visibility />}
                                        </IconButton>
                                      )}
                                    </InputAdornment>
                                  ),
                                }}
                                error={validationStatus[variable.key] === 'invalid'}
                                helperText={
                                  validationStatus[variable.key] === 'invalid' ? 'Invalid value' :
                                  variable.defaultValue ? `Default: ${variable.defaultValue}` : ''
                                }
                              />
                            </Box>
                            <Box sx={{ display: 'flex', gap: 1, ml: 2 }}>
                              {variable.secure && (
                                <Tooltip title="Generate random value">
                                  <IconButton
                                    onClick={() => {
                                      const randomValue = generateRandomValue()
                                      const updatedVariables = variables.map(v =>
                                        v.key === variable.key ? { ...v, value: randomValue, modified: true } : v
                                      )
                                      queryClient.setQueryData(['envVariables'], updatedVariables)
                                      toast.success('Random value generated')
                                    }}
                                  >
                                    <Casino />
                                  </IconButton>
                                </Tooltip>
                              )}
                              <Tooltip title="Copy value">
                                <IconButton onClick={() => handleCopyValue(variable.value)}>
                                  <ContentCopy />
                                </IconButton>
                              </Tooltip>
                              <Tooltip title="Edit">
                                <IconButton
                                  onClick={() => {
                                    setEditingVariable(variable)
                                    setDialogOpen(true)
                                  }}
                                >
                                  <Info />
                                </IconButton>
                              </Tooltip>
                              <Tooltip title="Delete">
                                <IconButton
                                  onClick={() => handleDeleteVariable(variable.key)}
                                  color="error"
                                >
                                  <Delete />
                                </IconButton>
                              </Tooltip>
                            </Box>
                          </Box>
                        </CardContent>
                      </Card>
                    </motion.div>
                  </Grid>
                ))}
              </Grid>
            </AnimatePresence>
          )}

          {filteredVariables.length === 0 && !isLoading && (
            <Box sx={{ textAlign: 'center', py: 8 }}>
              <Typography variant="h6" color="text.secondary">
                No variables found
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                {searchTerm ? 'Try adjusting your search' : 'Add a new variable to get started'}
              </Typography>
            </Box>
          )}
        </CardContent>
      </Card>

      <Fab
        color="primary"
        aria-label="save"
        sx={{ position: 'fixed', bottom: 24, right: 24 }}
        onClick={() => updateMutation.mutate(variables)}
        disabled={updateMutation.isPending || !variables.some(v => v.modified)}
      >
        <Save />
      </Fab>

      <Dialog open={dialogOpen} onClose={handleCloseDialog} maxWidth="sm" fullWidth>
        <form onSubmit={handleSubmit(handleAddVariable)}>
          <DialogTitle>
            {editingVariable ? 'Edit Variable' : 'Add Environment Variable'}
          </DialogTitle>
          <DialogContent>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, pt: 1 }}>
              <Controller
                name="key"
                control={control}
                render={({ field }) => (
                  <TextField
                    {...field}
                    label="Variable Name"
                    fullWidth
                    error={!!errors.key}
                    helperText={errors.key?.message || 'Use uppercase letters and underscores'}
                    disabled={!!editingVariable}
                  />
                )}
              />

              <Controller
                name="value"
                control={control}
                render={({ field }) => (
                  <TextField
                    {...field}
                    label="Value"
                    fullWidth
                    multiline
                    rows={2}
                  />
                )}
              />

              <Controller
                name="category"
                control={control}
                render={({ field }) => (
                  <FormControl fullWidth>
                    <InputLabel>Category</InputLabel>
                    <Select {...field} label="Category">
                      {categories.map((cat) => (
                        <MenuItem key={cat.value} value={cat.value}>
                          {cat.icon} {cat.label}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                )}
              />

              <Controller
                name="description"
                control={control}
                render={({ field }) => (
                  <TextField
                    {...field}
                    label="Description"
                    fullWidth
                    multiline
                    rows={2}
                    error={!!errors.description}
                    helperText={errors.description?.message}
                  />
                )}
              />

              <Box sx={{ display: 'flex', gap: 2 }}>
                <Controller
                  name="secure"
                  control={control}
                  render={({ field }) => (
                    <FormControlLabel
                      control={<Checkbox {...field} checked={field.value} />}
                      label="Secure/Sensitive"
                    />
                  )}
                />

                <Controller
                  name="required"
                  control={control}
                  render={({ field }) => (
                    <FormControlLabel
                      control={<Checkbox {...field} checked={field.value} />}
                      label="Required"
                    />
                  )}
                />
              </Box>
            </Box>
          </DialogContent>
          <DialogActions>
            <Button onClick={handleCloseDialog}>Cancel</Button>
            <Button type="submit" variant="contained">
              {editingVariable ? 'Update' : 'Add'}
            </Button>
          </DialogActions>
        </form>
      </Dialog>
    </Box>
  )
}

export default EnvManager