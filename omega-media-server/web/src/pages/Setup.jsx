import React, { useState } from 'react';
import { CheckCircleIcon } from '@heroicons/react/24/solid';
import { motion, AnimatePresence } from 'framer-motion';

const steps = [
  { id: 'welcome', title: 'Welcome', description: 'Welcome to Omega Media Server' },
  { id: 'admin', title: 'Admin Account', description: 'Create your administrator account' },
  { id: 'media', title: 'Media Paths', description: 'Configure your media directories' },
  { id: 'features', title: 'Features', description: 'Select features to enable' },
  { id: 'ai', title: 'AI Settings', description: 'Configure AI and ML features' },
  { id: 'complete', title: 'Complete', description: 'Setup is complete!' },
];

const Setup = ({ onComplete }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [formData, setFormData] = useState({
    adminEmail: '',
    adminPassword: '',
    adminPasswordConfirm: '',
    mediaPath: '/media',
    enableAI: true,
    enable8K: true,
    enableVPN: false,
    enablePlex: false,
    aiModel: 'advanced',
    recommendationEngine: 'collaborative',
    subtitleLanguages: ['en', 'es', 'fr'],
  });

  const handleNext = async () => {
    if (currentStep === steps.length - 1) {
      // Submit setup data
      try {
        const response = await fetch('/api/setup', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(formData),
        });
        
        if (response.ok) {
          onComplete();
        } else {
          console.error('Setup failed');
        }
      } catch (error) {
        console.error('Setup error:', error);
      }
    } else {
      setCurrentStep(currentStep + 1);
    }
  };

  const handleBack = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const updateFormData = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const renderStepContent = () => {
    switch (steps[currentStep].id) {
      case 'welcome':
        return (
          <div className="text-center space-y-6">
            <div className="w-32 h-32 mx-auto bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
              <span className="text-white text-5xl font-bold">Ω</span>
            </div>
            <h1 className="text-4xl font-bold">Welcome to Omega Media Server</h1>
            <p className="text-xl text-gray-600">
              The ultimate all-in-one media solution for 2025
            </p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-8">
              <div className="text-center">
                <div className="text-3xl mb-2">🎬</div>
                <p className="text-sm">30+ Apps</p>
              </div>
              <div className="text-center">
                <div className="text-3xl mb-2">🤖</div>
                <p className="text-sm">AI Powered</p>
              </div>
              <div className="text-center">
                <div className="text-3xl mb-2">🔒</div>
                <p className="text-sm">Secure</p>
              </div>
              <div className="text-center">
                <div className="text-3xl mb-2">⚡</div>
                <p className="text-sm">8K Ready</p>
              </div>
            </div>
          </div>
        );

      case 'admin':
        return (
          <div className="space-y-6">
            <h2 className="text-2xl font-semibold">Create Admin Account</h2>
            <div>
              <label className="block text-sm font-medium mb-2">Email Address</label>
              <input
                type="email"
                className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
                value={formData.adminEmail}
                onChange={(e) => updateFormData('adminEmail', e.target.value)}
                placeholder="admin@example.com"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">Password</label>
              <input
                type="password"
                className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
                value={formData.adminPassword}
                onChange={(e) => updateFormData('adminPassword', e.target.value)}
                placeholder="Enter a strong password"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">Confirm Password</label>
              <input
                type="password"
                className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
                value={formData.adminPasswordConfirm}
                onChange={(e) => updateFormData('adminPasswordConfirm', e.target.value)}
                placeholder="Confirm your password"
              />
            </div>
          </div>
        );

      case 'media':
        return (
          <div className="space-y-6">
            <h2 className="text-2xl font-semibold">Configure Media Directories</h2>
            <p className="text-gray-600">
              Omega will automatically organize your media into these directories.
            </p>
            <div className="space-y-4">
              <div className="bg-gray-50 p-4 rounded-lg">
                <label className="block text-sm font-medium mb-2">Base Media Path</label>
                <input
                  type="text"
                  className="w-full px-4 py-2 border rounded-lg"
                  value={formData.mediaPath}
                  onChange={(e) => updateFormData('mediaPath', e.target.value)}
                />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-gray-50 p-3 rounded">
                  <p className="font-medium">Movies</p>
                  <p className="text-sm text-gray-600">{formData.mediaPath}/movies</p>
                </div>
                <div className="bg-gray-50 p-3 rounded">
                  <p className="font-medium">TV Shows</p>
                  <p className="text-sm text-gray-600">{formData.mediaPath}/tv</p>
                </div>
                <div className="bg-gray-50 p-3 rounded">
                  <p className="font-medium">Music</p>
                  <p className="text-sm text-gray-600">{formData.mediaPath}/music</p>
                </div>
                <div className="bg-gray-50 p-3 rounded">
                  <p className="font-medium">Photos</p>
                  <p className="text-sm text-gray-600">{formData.mediaPath}/photos</p>
                </div>
              </div>
            </div>
          </div>
        );

      case 'features':
        return (
          <div className="space-y-6">
            <h2 className="text-2xl font-semibold">Select Features</h2>
            <div className="space-y-4">
              <label className="flex items-center space-x-3 p-4 border rounded-lg hover:bg-gray-50">
                <input
                  type="checkbox"
                  className="w-5 h-5 text-blue-600"
                  checked={formData.enableAI}
                  onChange={(e) => updateFormData('enableAI', e.target.checked)}
                />
                <div>
                  <p className="font-medium">AI & Machine Learning</p>
                  <p className="text-sm text-gray-600">Smart recommendations, auto-tagging, voice control</p>
                </div>
              </label>
              <label className="flex items-center space-x-3 p-4 border rounded-lg hover:bg-gray-50">
                <input
                  type="checkbox"
                  className="w-5 h-5 text-blue-600"
                  checked={formData.enable8K}
                  onChange={(e) => updateFormData('enable8K', e.target.checked)}
                />
                <div>
                  <p className="font-medium">8K Streaming Support</p>
                  <p className="text-sm text-gray-600">Enable 8K HDR video streaming</p>
                </div>
              </label>
              <label className="flex items-center space-x-3 p-4 border rounded-lg hover:bg-gray-50">
                <input
                  type="checkbox"
                  className="w-5 h-5 text-blue-600"
                  checked={formData.enableVPN}
                  onChange={(e) => updateFormData('enableVPN', e.target.checked)}
                />
                <div>
                  <p className="font-medium">Built-in VPN Server</p>
                  <p className="text-sm text-gray-600">WireGuard VPN for secure remote access</p>
                </div>
              </label>
              <label className="flex items-center space-x-3 p-4 border rounded-lg hover:bg-gray-50">
                <input
                  type="checkbox"
                  className="w-5 h-5 text-blue-600"
                  checked={formData.enablePlex}
                  onChange={(e) => updateFormData('enablePlex', e.target.checked)}
                />
                <div>
                  <p className="font-medium">Plex Media Server</p>
                  <p className="text-sm text-gray-600">Enable Plex alongside Jellyfin (requires Plex Pass for some features)</p>
                </div>
              </label>
            </div>
          </div>
        );

      case 'ai':
        return (
          <div className="space-y-6">
            <h2 className="text-2xl font-semibold">AI Configuration</h2>
            <div>
              <label className="block text-sm font-medium mb-2">AI Model</label>
              <select
                className="w-full px-4 py-2 border rounded-lg"
                value={formData.aiModel}
                onChange={(e) => updateFormData('aiModel', e.target.value)}
              >
                <option value="basic">Basic (Faster, less accurate)</option>
                <option value="advanced">Advanced (Recommended)</option>
                <option value="premium">Premium (Best accuracy, more resources)</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">Recommendation Engine</label>
              <select
                className="w-full px-4 py-2 border rounded-lg"
                value={formData.recommendationEngine}
                onChange={(e) => updateFormData('recommendationEngine', e.target.value)}
              >
                <option value="basic">Basic (Content-based)</option>
                <option value="collaborative">Collaborative Filtering</option>
                <option value="hybrid">Hybrid (Best results)</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">Auto-Subtitle Languages</label>
              <p className="text-sm text-gray-600 mb-2">
                Select languages for automatic subtitle generation
              </p>
              <div className="grid grid-cols-3 gap-2">
                {['en', 'es', 'fr', 'de', 'ja', 'ko', 'zh', 'it', 'pt'].map(lang => (
                  <label key={lang} className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      checked={formData.subtitleLanguages.includes(lang)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          updateFormData('subtitleLanguages', [...formData.subtitleLanguages, lang]);
                        } else {
                          updateFormData('subtitleLanguages', formData.subtitleLanguages.filter(l => l !== lang));
                        }
                      }}
                    />
                    <span className="text-sm">{lang.toUpperCase()}</span>
                  </label>
                ))}
              </div>
            </div>
          </div>
        );

      case 'complete':
        return (
          <div className="text-center space-y-6">
            <div className="w-24 h-24 mx-auto bg-green-100 rounded-full flex items-center justify-center">
              <CheckCircleIcon className="w-16 h-16 text-green-500" />
            </div>
            <h2 className="text-3xl font-bold">Setup Complete!</h2>
            <p className="text-xl text-gray-600">
              Omega Media Server is ready to use
            </p>
            <div className="bg-gray-50 p-6 rounded-lg text-left max-w-md mx-auto">
              <h3 className="font-semibold mb-3">Quick Start Guide:</h3>
              <ul className="space-y-2 text-sm">
                <li>✓ Access the dashboard at <code className="bg-gray-200 px-1 rounded">http://localhost</code></li>
                <li>✓ Login with your admin account</li>
                <li>✓ Add media to your configured directories</li>
                <li>✓ Install additional apps from the Apps page</li>
                <li>✓ Invite users from the Users page</li>
              </ul>
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-purple-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl shadow-xl max-w-4xl w-full p-8">
        {/* Progress Bar */}
        <div className="mb-8">
          <div className="flex justify-between mb-2">
            {steps.map((step, index) => (
              <div
                key={step.id}
                className={`flex items-center ${index !== steps.length - 1 ? 'flex-1' : ''}`}
              >
                <div
                  className={`w-10 h-10 rounded-full flex items-center justify-center text-sm font-medium
                    ${index <= currentStep
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-200 text-gray-600'
                    }`}
                >
                  {index < currentStep ? '✓' : index + 1}
                </div>
                {index !== steps.length - 1 && (
                  <div
                    className={`flex-1 h-1 mx-2 ${
                      index < currentStep ? 'bg-blue-600' : 'bg-gray-200'
                    }`}
                  />
                )}
              </div>
            ))}
          </div>
          <div className="text-center">
            <p className="text-sm text-gray-600">{steps[currentStep].description}</p>
          </div>
        </div>

        {/* Step Content */}
        <AnimatePresence mode="wait">
          <motion.div
            key={currentStep}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.3 }}
            className="mb-8"
          >
            {renderStepContent()}
          </motion.div>
        </AnimatePresence>

        {/* Navigation Buttons */}
        <div className="flex justify-between">
          <button
            onClick={handleBack}
            disabled={currentStep === 0}
            className={`px-6 py-2 rounded-lg font-medium
              ${currentStep === 0
                ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
          >
            Back
          </button>
          <button
            onClick={handleNext}
            className="px-6 py-2 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700"
          >
            {currentStep === steps.length - 1 ? 'Complete Setup' : 'Next'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default Setup;