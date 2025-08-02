import React, { useState } from 'react'
import {
  Box,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  Button,
  Typography,
  Card,
  CardContent,
  Paper,
  Chip,
  LinearProgress,
  Alert,
  IconButton,
  Collapse,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  Grid,
  TextField,
  FormControlLabel,
  Checkbox,
} from '@mui/material'
import {
  PlayArrow,
  CheckCircle,
  Info,
  Warning,
  Code,
  Settings,
  Cloud,
  Security,
  Speed,
  Storage,
  NavigateNext,
  NavigateBefore,
  Close,
  ExpandMore,
  ExpandLess,
  ContentCopy,
  Terminal,
  GitHub,
} from '@mui/icons-material'
import { motion, AnimatePresence } from 'framer-motion'
import Joyride, { CallBackProps, STATUS, Step as JoyrideStep } from 'react-joyride'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { atomDark } from 'react-syntax-highlighter/dist/esm/styles/prism'
import toast from 'react-hot-toast'

interface TutorialSection {
  title: string
  description: string
  icon: React.ReactNode
  steps: TutorialStep[]
}

interface TutorialStep {
  title: string
  content: string
  code?: string
  language?: string
  tips?: string[]
  warning?: string
  completed?: boolean
}

const tutorialSections: TutorialSection[] = [
  {
    title: 'Getting Started',
    description: 'Learn the basics of your media server',
    icon: <PlayArrow />,
    steps: [
      {
        title: 'Welcome to Your Media Server',
        content: 'This powerful media server brings together all your favorite streaming services in one place. Let\'s get you set up!',
        tips: [
          'The dashboard provides real-time monitoring of all services',
          'You can control services directly from the interface',
          'Environment variables can be managed through the UI',
        ],
      },
      {
        title: 'Understanding the Architecture',
        content: 'Your media server uses Docker containers to run isolated services that work together seamlessly.',
        code: `# Core services include:
- Jellyfin: Your personal Netflix
- Sonarr/Radarr: Automated media management
- qBittorrent: Download client
- Prowlarr: Indexer management
- And many more!`,
        language: 'yaml',
      },
      {
        title: 'Initial Configuration',
        content: 'Let\'s configure your basic settings to get started.',
        code: `# Set your timezone
TZ=America/New_York

# Configure user permissions
PUID=1000
PGID=1000

# Set your domain (optional)
DOMAIN=media.yourdomain.com`,
        language: 'bash',
        warning: 'Make sure to use your actual timezone and user IDs',
      },
    ],
  },
  {
    title: 'Service Management',
    description: 'Control and monitor your media services',
    icon: <Settings />,
    steps: [
      {
        title: 'Starting Services',
        content: 'Services can be started individually or all at once using the dashboard controls.',
        code: `# Start all services
docker-compose up -d

# Start specific service
docker-compose up -d jellyfin

# View service logs
docker-compose logs -f jellyfin`,
        language: 'bash',
        tips: [
          'Services start automatically on system boot',
          'Health checks ensure services stay running',
          'Logs are accessible through the UI',
        ],
      },
      {
        title: 'Service Dependencies',
        content: 'Some services depend on others. The system manages these dependencies automatically.',
        code: `depends_on:
  - redis
  - postgres
  - traefik`,
        language: 'yaml',
      },
      {
        title: 'Monitoring Service Health',
        content: 'Each service has health checks that monitor its status and automatically restart if needed.',
        tips: [
          'Green status means healthy',
          'Yellow means starting or warning',
          'Red indicates an error',
        ],
      },
    ],
  },
  {
    title: 'Media Organization',
    description: 'Set up your media library structure',
    icon: <Storage />,
    steps: [
      {
        title: 'Directory Structure',
        content: 'Organize your media files for optimal performance and compatibility.',
        code: `/media
├── movies
│   ├── Movie Name (Year)
│   │   └── movie.mkv
├── tv
│   ├── Show Name
│   │   ├── Season 01
│   │   │   └── S01E01.mkv
├── music
│   ├── Artist
│   │   └── Album
│   │       └── 01-track.mp3
└── books
    └── Author
        └── Book.epub`,
        language: 'bash',
      },
      {
        title: 'Permissions',
        content: 'Ensure proper file permissions for media access.',
        code: `# Set ownership
sudo chown -R $USER:$USER /media

# Set permissions
sudo chmod -R 755 /media`,
        language: 'bash',
        warning: 'Incorrect permissions can prevent services from accessing media',
      },
    ],
  },
  {
    title: 'Automation',
    description: 'Set up automated media management',
    icon: <Speed />,
    steps: [
      {
        title: 'Configuring Sonarr',
        content: 'Sonarr automates TV show downloads and organization.',
        tips: [
          'Add your indexers in Prowlarr first',
          'Configure quality profiles based on your preferences',
          'Set up download clients',
          'Add shows to start monitoring',
        ],
      },
      {
        title: 'Configuring Radarr',
        content: 'Radarr handles movie automation similar to Sonarr.',
        tips: [
          'Import existing movies for monitoring',
          'Set up lists for automatic additions',
          'Configure custom formats for quality',
        ],
      },
      {
        title: 'Download Client Setup',
        content: 'Configure qBittorrent for optimal performance.',
        code: `# Recommended settings:
- Enable Web UI
- Set download directory
- Configure connection limits
- Enable encryption
- Set up categories`,
        language: 'yaml',
      },
    ],
  },
  {
    title: 'Security',
    description: 'Secure your media server',
    icon: <Security />,
    steps: [
      {
        title: 'Basic Security',
        content: 'Implement essential security measures.',
        code: `# Use strong passwords
# Enable 2FA where available
# Keep services updated
# Use HTTPS with Traefik
# Configure firewall rules`,
        language: 'bash',
        warning: 'Never expose services directly to the internet without protection',
      },
      {
        title: 'VPN Configuration',
        content: 'Route download traffic through VPN for privacy.',
        tips: [
          'Configure VPN container',
          'Route qBittorrent through VPN',
          'Set up kill switch',
          'Test for DNS leaks',
        ],
      },
    ],
  },
  {
    title: 'Advanced Features',
    description: 'Unlock the full potential',
    icon: <Cloud />,
    steps: [
      {
        title: 'Remote Access',
        content: 'Access your media server from anywhere.',
        code: `# Using Traefik reverse proxy
# Configure DNS records
# Set up SSL certificates
# Enable port forwarding (if needed)`,
        language: 'bash',
      },
      {
        title: 'Mobile Apps',
        content: 'Connect mobile apps for on-the-go access.',
        tips: [
          'Jellyfin has iOS and Android apps',
          'Overseerr works great on mobile',
          'Most services have mobile-friendly web UIs',
        ],
      },
      {
        title: 'Transcoding',
        content: 'Enable hardware transcoding for better performance.',
        code: `# Intel Quick Sync
devices:
  - /dev/dri:/dev/dri

# NVIDIA GPU
runtime: nvidia
environment:
  - NVIDIA_VISIBLE_DEVICES=all`,
        language: 'yaml',
      },
    ],
  },
]

const joyrideSteps: JoyrideStep[] = [
  {
    target: '.tutorial-sections',
    content: 'Choose a tutorial section to begin learning about your media server.',
    placement: 'center',
  },
  {
    target: '.tutorial-progress',
    content: 'Track your progress through each tutorial section.',
  },
  {
    target: '.tutorial-tips',
    content: 'Find helpful tips and best practices here.',
  },
  {
    target: '.tutorial-code',
    content: 'Copy code snippets directly to your clipboard.',
  },
]

const Tutorial: React.FC = () => {
  const [activeSection, setActiveSection] = useState(0)
  const [activeStep, setActiveStep] = useState(0)
  const [completedSteps, setCompletedSteps] = useState<{ [key: string]: boolean }>({})
  const [runTour, setRunTour] = useState(false)
  const [expandedTips, setExpandedTips] = useState(true)
  const [userNotes, setUserNotes] = useState('')

  const currentSection = tutorialSections[activeSection]
  const currentStep = currentSection.steps[activeStep]
  const progress = (Object.keys(completedSteps).length / tutorialSections.reduce((acc, section) => acc + section.steps.length, 0)) * 100

  const handleStepComplete = () => {
    const stepKey = `${activeSection}-${activeStep}`
    setCompletedSteps({ ...completedSteps, [stepKey]: true })
    
    if (activeStep < currentSection.steps.length - 1) {
      setActiveStep(activeStep + 1)
    } else if (activeSection < tutorialSections.length - 1) {
      setActiveSection(activeSection + 1)
      setActiveStep(0)
    } else {
      toast.success('Congratulations! You\'ve completed all tutorials!')
    }
  }

  const handleJoyrideCallback = (data: CallBackProps) => {
    const { status } = data
    if ([STATUS.FINISHED, STATUS.SKIPPED].includes(status)) {
      setRunTour(false)
    }
  }

  const copyCode = (code: string) => {
    navigator.clipboard.writeText(code)
    toast.success('Code copied to clipboard')
  }

  return (
    <Box>
      <Joyride
        steps={joyrideSteps}
        run={runTour}
        continuous
        showSkipButton
        styles={{
          options: {
            primaryColor: '#00ff88',
            zIndex: 10000,
          },
        }}
        callback={handleJoyrideCallback}
      />

      <Box sx={{ mb: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Box>
          <Typography variant="h4" gutterBottom>
            Interactive Tutorial
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Learn how to use your media server like a pro
          </Typography>
        </Box>
        <Button
          variant="outlined"
          startIcon={<Info />}
          onClick={() => setRunTour(true)}
        >
          Start Tour
        </Button>
      </Box>

      <Box className="tutorial-progress" sx={{ mb: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
          <Typography variant="body2">Overall Progress</Typography>
          <Typography variant="body2">{Math.round(progress)}%</Typography>
        </Box>
        <LinearProgress
          variant="determinate"
          value={progress}
          sx={{
            height: 8,
            borderRadius: 4,
            backgroundColor: 'rgba(255, 255, 255, 0.1)',
            '& .MuiLinearProgress-bar': {
              borderRadius: 4,
              background: 'linear-gradient(90deg, #00ff88 0%, #00cc66 100%)',
            },
          }}
        />
      </Box>

      <Grid container spacing={3}>
        <Grid item xs={12} md={3}>
          <Card className="tutorial-sections">
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Sections
              </Typography>
              <List>
                {tutorialSections.map((section, index) => (
                  <React.Fragment key={index}>
                    <ListItem
                      button
                      selected={activeSection === index}
                      onClick={() => {
                        setActiveSection(index)
                        setActiveStep(0)
                      }}
                    >
                      <ListItemIcon>{section.icon}</ListItemIcon>
                      <ListItemText
                        primary={section.title}
                        secondary={section.description}
                      />
                    </ListItem>
                    {index < tutorialSections.length - 1 && <Divider />}
                  </React.Fragment>
                ))}
              </List>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={9}>
          <AnimatePresence mode="wait">
            <motion.div
              key={`${activeSection}-${activeStep}`}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.3 }}
            >
              <Card>
                <CardContent>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                    <Typography variant="h5">
                      {currentStep.title}
                    </Typography>
                    <Chip
                      label={`Step ${activeStep + 1} of ${currentSection.steps.length}`}
                      color="primary"
                      size="small"
                    />
                  </Box>

                  <Typography variant="body1" paragraph>
                    {currentStep.content}
                  </Typography>

                  {currentStep.warning && (
                    <Alert severity="warning" sx={{ mb: 2 }}>
                      {currentStep.warning}
                    </Alert>
                  )}

                  {currentStep.code && (
                    <Box className="tutorial-code" sx={{ mb: 3, position: 'relative' }}>
                      <IconButton
                        size="small"
                        onClick={() => copyCode(currentStep.code!)}
                        sx={{
                          position: 'absolute',
                          top: 8,
                          right: 8,
                          zIndex: 1,
                          backgroundColor: 'rgba(0, 0, 0, 0.5)',
                          '&:hover': {
                            backgroundColor: 'rgba(0, 0, 0, 0.7)',
                          },
                        }}
                      >
                        <ContentCopy fontSize="small" />
                      </IconButton>
                      <SyntaxHighlighter
                        language={currentStep.language || 'bash'}
                        style={atomDark}
                        customStyle={{
                          borderRadius: 8,
                          fontSize: '0.9rem',
                        }}
                      >
                        {currentStep.code}
                      </SyntaxHighlighter>
                    </Box>
                  )}

                  {currentStep.tips && currentStep.tips.length > 0 && (
                    <Box className="tutorial-tips" sx={{ mb: 3 }}>
                      <Box
                        sx={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center',
                          mb: 1,
                          cursor: 'pointer',
                        }}
                        onClick={() => setExpandedTips(!expandedTips)}
                      >
                        <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Info color="primary" />
                          Tips & Tricks
                        </Typography>
                        {expandedTips ? <ExpandLess /> : <ExpandMore />}
                      </Box>
                      <Collapse in={expandedTips}>
                        <Paper sx={{ p: 2, backgroundColor: 'rgba(0, 255, 136, 0.05)' }}>
                          <List dense>
                            {currentStep.tips.map((tip, index) => (
                              <ListItem key={index}>
                                <ListItemIcon>
                                  <CheckCircle color="primary" fontSize="small" />
                                </ListItemIcon>
                                <ListItemText primary={tip} />
                              </ListItem>
                            ))}
                          </List>
                        </Paper>
                      </Collapse>
                    </Box>
                  )}

                  <Divider sx={{ my: 3 }} />

                  <Box sx={{ mb: 3 }}>
                    <Typography variant="subtitle1" gutterBottom>
                      Your Notes
                    </Typography>
                    <TextField
                      multiline
                      rows={3}
                      fullWidth
                      placeholder="Add your personal notes here..."
                      value={userNotes}
                      onChange={(e) => setUserNotes(e.target.value)}
                      variant="outlined"
                    />
                  </Box>

                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Box sx={{ display: 'flex', gap: 1 }}>
                      <Button
                        variant="outlined"
                        startIcon={<NavigateBefore />}
                        onClick={() => {
                          if (activeStep > 0) {
                            setActiveStep(activeStep - 1)
                          } else if (activeSection > 0) {
                            setActiveSection(activeSection - 1)
                            setActiveStep(tutorialSections[activeSection - 1].steps.length - 1)
                          }
                        }}
                        disabled={activeSection === 0 && activeStep === 0}
                      >
                        Previous
                      </Button>
                      <Button
                        variant="contained"
                        endIcon={<NavigateNext />}
                        onClick={handleStepComplete}
                      >
                        {activeStep === currentSection.steps.length - 1 && activeSection === tutorialSections.length - 1
                          ? 'Complete'
                          : 'Next'}
                      </Button>
                    </Box>

                    <FormControlLabel
                      control={
                        <Checkbox
                          checked={completedSteps[`${activeSection}-${activeStep}`] || false}
                          onChange={(e) => {
                            const stepKey = `${activeSection}-${activeStep}`
                            setCompletedSteps({
                              ...completedSteps,
                              [stepKey]: e.target.checked,
                            })
                          }}
                        />
                      }
                      label="Mark as completed"
                    />
                  </Box>
                </CardContent>
              </Card>
            </motion.div>
          </AnimatePresence>

          <Box sx={{ mt: 3, display: 'flex', gap: 2 }}>
            <Card sx={{ flex: 1 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Quick Links
                </Typography>
                <List dense>
                  <ListItem button component="a" href="https://jellyfin.org/docs/" target="_blank">
                    <ListItemIcon><Terminal /></ListItemIcon>
                    <ListItemText primary="Jellyfin Documentation" />
                  </ListItem>
                  <ListItem button component="a" href="https://wiki.servarr.com/" target="_blank">
                    <ListItemIcon><Code /></ListItemIcon>
                    <ListItemText primary="Servarr Wiki" />
                  </ListItem>
                  <ListItem button component="a" href="https://github.com" target="_blank">
                    <ListItemIcon><GitHub /></ListItemIcon>
                    <ListItemText primary="GitHub Repository" />
                  </ListItem>
                </List>
              </CardContent>
            </Card>

            <Card sx={{ flex: 1 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Need Help?
                </Typography>
                <Typography variant="body2" paragraph>
                  Join our community for support and discussions.
                </Typography>
                <Button variant="outlined" fullWidth startIcon={<Info />}>
                  Visit Support Forum
                </Button>
              </CardContent>
            </Card>
          </Box>
        </Grid>
      </Grid>
    </Box>
  )
}

export default Tutorial