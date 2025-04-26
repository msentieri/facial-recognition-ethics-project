<template>
  <div class="min-h-screen bg-gradient-to-br from-blue-50 to-purple-50 text-gray-800">
    <header class="bg-indigo-600 text-white shadow-lg">
      <div class="container mx-auto px-6 py-8">
        <h1 class="text-4xl font-bold mb-2">Facial Recognition Technology</h1>
        <p class="text-xl text-indigo-100">Ethics, Risks, and Policy Recommendations</p>
      </div>
    </header>

    <main class="container mx-auto px-6 py-8">
      <div class="bg-white rounded-xl shadow-lg overflow-hidden mb-8">
        <div class="flex border-b border-gray-200">
          <button 
            v-for="(tab, index) in tabs" 
            :key="index"
            @click="activeTab = index"
            :class="{'bg-indigo-100 text-indigo-700 font-semibold': activeTab === index, 'hover:bg-gray-50': activeTab !== index}"
            class="px-4 py-3 text-sm font-medium focus:outline-none transition-colors"
          >
            {{ tab }}
          </button>
        </div>

        <div class="p-6">
          <!-- Overview Tab -->
          <div v-if="activeTab === 0" class="fade-in">
            <div class="grid md:grid-cols-2 gap-8">
              <div>
                <h2 class="text-2xl font-bold mb-4 text-indigo-700">Understanding Facial Recognition</h2>
                <p class="mb-4">Facial recognition technology (FRT) uses biometric data to identify individuals by analyzing facial features. While offering potential benefits for security and convenience, its rapid adoption has outpaced regulatory frameworks and ethical considerations.</p>
                
                <div class="bg-yellow-50 border-l-4 border-yellow-400 p-4 mb-6">
                  <div class="flex items-start">
                    <div class="flex-shrink-0 pt-0.5">
                      <svg class="h-5 w-5 text-yellow-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                        <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd" />
                      </svg>
                    </div>
                    <div class="ml-3">
                      <h3 class="text-sm font-medium text-yellow-800">Key Concern</h3>
                      <div class="mt-1 text-sm text-yellow-700">
                        <p>Over 60% of Americans believe facial recognition use by companies and government should be strictly limited (Pew Research Center, 2019).</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
              
              <div class="bg-indigo-50 p-4 rounded-lg">
                <h3 class="text-lg font-semibold mb-3 text-indigo-700">Quick Facts</h3>
                <ul class="space-y-2">
                  <li class="flex items-start">
                    <svg class="h-4 w-4 text-indigo-500 mt-1 mr-2 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                    </svg>
                    <span class="text-sm">FRT market projected to reach $12.67 billion by 2028 (Grand View Research)</span>
                  </li>
                  <li class="flex items-start">
                    <svg class="h-4 w-4 text-indigo-500 mt-1 mr-2 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                    </svg>
                    <span class="text-sm">Black and Asian faces up to 100 times more likely to be misidentified (NIST study)</span>
                  </li>
                  <li class="flex items-start">
                    <svg class="h-4 w-4 text-indigo-500 mt-1 mr-2 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                    </svg>
                    <span class="text-sm">Several US cities have banned government use of FRT</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>

          <!-- Ethical Issues Tab -->
          <div v-if="activeTab === 1" class="fade-in">
            <h2 class="text-2xl font-bold mb-6 text-indigo-700">Ethical Concerns</h2>
            
            <div class="grid md:grid-cols-2 gap-6">
              <div>
                <h3 class="text-lg font-semibold mb-3 flex items-center">
                  <span class="w-6 h-6 bg-red-100 text-red-600 rounded-full flex items-center justify-center mr-2 text-sm">1</span>
                  Bias and Inaccuracy
                </h3>
                <p class="mb-3 text-sm">Multiple studies have demonstrated significant racial and gender bias in facial recognition systems:</p>
                <ul class="list-disc pl-5 space-y-1 mb-4 text-sm">
                  <li>NIST found highest false positive rates for African American and Asian faces (up to 100 times higher than white faces) <span class="text-xs text-gray-500">(NIST, 2019)</span></li>
                  <li>Women more likely to be misidentified than men <span class="text-xs text-gray-500">(Buolamwini & Gebru, 2018)</span></li>
                  <li>Higher error rates for darker-skinned individuals <span class="text-xs text-gray-500">(Rajput et al., 2020)</span></li>
                </ul>
                
                <h3 class="text-lg font-semibold mb-3 flex items-center">
                  <span class="w-6 h-6 bg-red-100 text-red-600 rounded-full flex items-center justify-center mr-2 text-sm">2</span>
                  Privacy Violations
                </h3>
                <p class="mb-3 text-sm">FRT enables unprecedented mass surveillance capabilities:</p>
                <ul class="list-disc pl-5 space-y-1 text-sm">
                  <li>Most individuals unaware when their biometric data is collected <span class="text-xs text-gray-500">(EPIC, 2021)</span></li>
                  <li>Lack of consent in public spaces and commercial settings</li>
                  <li>Potential for "perpetual line-ups" with no expiration of biometric data</li>
                </ul>
              </div>
              
              <div>
                <h3 class="text-lg font-semibold mb-3 flex items-center">
                  <span class="w-6 h-6 bg-red-100 text-red-600 rounded-full flex items-center justify-center mr-2 text-sm">3</span>
                  Chilling Effects on Society
                </h3>
                <p class="mb-3 text-sm">The presence of facial recognition can alter behavior and limit freedoms:</p>
                <ul class="list-disc pl-5 space-y-1 mb-4 text-sm">
                  <li>Reduces willingness to participate in protests <span class="text-xs text-gray-500">(Feldstein, 2019)</span></li>
                  <li>Creates "self-censorship" in public spaces</li>
                  <li>Dispro portionately impacts marginalized communities</li>
                </ul>
                
                <h3 class="text-lg font-semibold mb-3 flex items-center">
                  <span class="w-6 h-6 bg-red-100 text-red-600 rounded-full flex items-center justify-center mr-2 text-sm">4</span>
                  Misuse Potential
                </h3>
                <p class="mb-3 text-sm">Without proper safeguards, FRT can be abused:</p>
                <ul class="list-disc pl-5 space-y-1 text-sm">
                  <li>Stalkers using commercial FRT to track individuals <span class="text-xs text-gray-500">(NYT, 2020)</span></li>
                  <li>Authoritarian governments targeting dissidents <span class="text-xs text-gray-500">(Amnesty International, 2021)</span></li>
                  <li>Law enforcement using unreliable matches for arrests <span class="text-xs text-gray-500">(ACLU, 2020)</span></li>
                </ul>
              </div>
            </div>
          </div>

          <!-- MN Legislation Tab -->
          <div v-if="activeTab === 2" class="fade-in">
            <h2 class="text-2xl font-bold mb-6 text-indigo-700">Minnesota Policy Recommendations</h2>
            
            <div class="bg-blue-50 border-l-4 border-blue-400 p-3 mb-6">
              <div class="flex items-start">
                <div class="flex-shrink-0 pt-0.5">
                  <svg class="h-5 w-5 text-blue-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                    <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clip-rule="evenodd" />
                  </svg>
                </div>
                <div class="ml-3">
                  <h3 class="text-sm font-medium text-blue-800">Current Minnesota Status</h3>
                  <div class="mt-1 text-sm text-blue-700">
                    <p>As of 2023, Minnesota has no comprehensive laws regulating facial recognition technology use by government or private entities.</p>
                  </div>
                </div>
              </div>
            </div>
            
            <div class="grid md:grid-cols-2 gap-6">
              <div>
                <h3 class="text-lg font-semibold mb-4 text-indigo-600">Immediate Legislative Actions</h3>
                
                <div class="mb-4 p-3 bg-white border rounded-lg shadow-sm">
                  <h4 class="font-semibold mb-1 text-sm flex items-center">
                    <svg class="h-4 w-4 text-green-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                    </svg>
                    Warrant Requirement
                  </h4>
                  <p class="text-sm">Mandate judicial warrants for all law enforcement use of FRT, similar to requirements for wiretaps. Exceptions only for imminent threat to life.</p>
                </div>
                
                <div class="mb-4 p-3 bg-white border rounded-lg shadow-sm">
                  <h4 class="font-semibold mb-1 text-sm flex items-center">
                    <svg class="h-4 w-4 text-green-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                    </svg>
                    Prohibited Uses
                  </h4>
                  <p class="text-sm">Ban FRT in sensitive locations: schools, public housing, places of worship, and protests. Prohibit use for immigration enforcement.</p>
                </div>
                
                <div class="p-3 bg-white border rounded-lg shadow-sm">
                  <h4 class="font-semibold mb-1 text-sm flex items-center">
                    <svg class="h-4 w-4 text-green-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                    </svg>
                    Transparency Mandates
                  </h4>
                  <p class="text-sm">Require public reporting on FRT usage statistics, accuracy rates by demographic, and audit trails for all searches.</p>
                </div>
              </div>
              
              <div>
                <h3 class="text-lg font-semibold mb-4 text-indigo-600">Long-Term Policy Framework</h3>
                
                <div class="mb-4 p-3 bg-white border rounded-lg shadow-sm">
                  <h4 class="font-semibold mb-1 text-sm flex items-center">
                    <svg class="h-4 w-4 text-purple-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4"></path>
                    </svg>
                    Oversight Board
                  </h4>
                  <p class="text-sm">Establish an independent Facial Recognition Oversight Board with civil rights experts, technologists, and community representatives.</p>
                </div>
                
                <div class="mb-4 p-3 bg-white border rounded-lg shadow-sm">
                  <h4 class="font-semibold mb-1 text-sm flex items-center">
                    <svg class="h-4 w-4 text-purple-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4"></path>
                    </svg>
                    Private Sector Regulations
                  </h4>
                  <p class="text-sm">Require opt-in consent for commercial uses, data minimization standards, and right-to-know about FRT deployments in physical spaces.</p>
                </div>
                
                <div class="p-3 bg-white border rounded-lg shadow-sm">
                  <h4 class="font-semibold mb-1 text-sm flex items-center">
                    <svg class="h-4 w-4 text-purple-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 ```vue
0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4"></path>
                    </svg>
                    Accountability Measures
                  </h4>
                  <p class="text-sm">Create civil remedies for violations, including statutory damages and attorney fee provisions to enable enforcement.</p>
                </div>
              </div>
            </div>
          </div>

          <!-- Try It Tab -->
          <div v-if="activeTab === 3" class="fade-in">
            <div class="grid md:grid-cols-2 gap-6">
              <div>
                <h2 class="text-2xl font-bold mb-4 text-indigo-700">Facial Recognition Demo</h2>
                <p class="mb-4 text-sm">This demonstration shows how basic facial recognition works. Upload two images to compare faces.</p>
                
                <div class="bg-indigo-50 p-4 rounded-lg mb-4">
                  <h3 class="text-base font-semibold mb-2 text-indigo-700">How It Works</h3>
                  <ul class="space-y-2">
                    <li class="flex items-start">
                      <svg class="h-4 w-4 text-indigo-500 mt-0.5 mr-2 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                      </svg>
                      <span class="text-sm">Images are uploaded to server - processing does not happen locally</span>
                    </li>
                    <li class="flex items-start">
                      <svg class="h-4 w-4 text-indigo-500 mt-0.5 mr-2 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                      </svg>
                      <span class="text-sm">Demo uses DeepFace library for facial verification</span>
                    </li>
                    <li class="flex items-start">
                      <svg class="h-4 w-4 text-indigo-500 mt-0.5 mr-2 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                      </svg>
                      <span class="text-sm">Compares two faces and returns similarity score</span>
                    </li>
                  </ul>
                </div>
              </div>
              
              <div class="bg-white p-4 rounded-lg shadow-md">
                <!-- First Image Upload -->
                <div class="mb-4">
                  <h3 class="text-sm font-medium text-gray-700 mb-2">First Image</h3>
                  <div class="mt-1 flex justify-center px-6 pt-5 pb-6 border-2 border-gray-300 border-dashed rounded-md">
                    <div class="space-y-1 text-center">
                      <svg class="mx-auto h-8 w-8 text-gray-400" stroke="currentColor" fill="none" viewBox="0 0 48 48" aria-hidden="true">
                        <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v 4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
                      </svg>
                      <div class="flex text-sm text-gray-600 justify-center">
                        <button 
                          class="px-4 py-2 bg-indigo-600 text-white font-semibold rounded-md hover:bg-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-400"
                          @click="$refs.fileUpload1.click()"
                        >
                          Upload First Image
                        </button>
                        <input 
                          id="file-upload1" 
                          ref="fileUpload1"
                          name="file-upload1" 
                          type="file" 
                          @change="handleImageUpload($event, 'img1')" 
                          accept="image/*" 
                          class="hidden" 
                        />
                        <p class="pl-1 my-auto">or drag and drop</p>
                      </div>
                      <p class="text-xs text-gray-500">PNG, JPG up to 5MB</p>
                    </div>
                  </div>
                  <div v-if="uploadedImages.img1" class="mt-2">
                    <img :src="uploadedImages.img1" alt="First image preview" class="max-w-full h-auto rounded border border-gray-200 max-h-40" />
                  </div>
                </div>
                
                <!-- Second Image Upload -->
                <div class="mb-4">
                  <h3 class="text-sm font-medium text-gray-700 mb-2">Second Image</h3>
                  <div class="mt-1 flex justify-center px-6 pt-5 pb-6 border-2 border-gray-300 border-dashed rounded-md">
                    <div class="space-y-1 text-center">
                      <svg class="mx-auto h-8 w-8 text-gray-400" stroke="currentColor" fill="none" viewBox="0 0 48 48" aria-hidden="true">
                        <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
                      </svg>
                      <div class="flex text-sm text-gray-600 justify-center">
                        <button 
                          class="px-4 py-2 bg-indigo-600 text-white font-semibold rounded-md hover:bg-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-400"
                          @click="$refs.fileUpload2.click()"
                        >
                          Upload Second Image
                        </button>
                        <input 
                          id="file-upload2" 
                          ref="fileUpload2"
                          name="file-upload2" 
                          type="file" 
                          @change="handleImageUpload($event, 'img2')" 
                          accept="image/*" 
                          class="hidden" 
                        />
                        <p class="pl-1 my-auto">or drag and drop</p>
                      </div>
                      <p class="text-xs text-gray-500">PNG, JPG up to 5MB</p>
                    </div>
                  </div>
                  <div v-if="uploadedImages.img2" class="mt-2">
                    <img :src="uploadedImages.img2" alt="Second image preview" class="max-w-full h-auto rounded border border-gray-200 max-h-40" />
                  </div>
                </div>
                
                <!-- Compare Button -->
                <div class="flex justify-center mb-4">
                  <button 
                    @click="compareFaces" 
                    :disabled="!uploadedImages.img1 || !uploadedImages.img2 || isLoading"
                    :class="{'bg-indigo-400 cursor-not-allowed': !uploadedImages.img1 || !uploadedImages.img2, 'bg-indigo-600 hover:bg-indigo-500': uploadedImages.img1 && uploadedImages.img2}"
                    class="px-6 py-2 text-white font-semibold rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-400 transition-colors"
                  >
                    Compare Faces
                  </button>
                </div>
                
                <!-- Results Display -->
                <div v-if="result" class="mt-3">
                  <h3 class="text-sm font-semibold mb-1">Comparison Results</h3>
                  <div class="bg-gray-50 p-3 rounded text-xs">
                    <div v-if="result.error" class="text-red-600">{{ result.error }}</div>
                    <template v-else>
                      <div class="mb-2">
                        <span class="font-medium">Match Result: </span>
                        <span :class="{'text-green-600': result.verified, 'text-red-600': !result.verified}">
                          {{ result.verified ? 'Match Found' : 'No Match' }}
                        </span>
                      </div>
                      <div class="mb-2">
                        <span class="font-medium">Image Quality: </span>
                        <span>{{ result.confidence.toFixed(2) }}%</span>
                      </div>
                      <div class="mb-2">
                        <span class="font-medium">Threshold: </span>
                        <span>{{ result.threshold }}%</span>
                      </div>
                      <div class="mb-2">
                        <span class="font-medium">Model: </span>
                        <span>{{ result.model }}</span>
                      </div>
                    </template>
                  </div>
                </div>
                
                <div v-if="isLoading" class="mt-3">
                  <div class="flex items-center justify-center">
                    <svg class="animate-spin -ml-1 mr-2 h-4 w-4 text-indigo-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                      <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    <span class="text-sm">Processing images...</span>
                  </div>
                </div>
              </div>
            </div>
          </div>     

          <!-- Resources Tab -->
          <div v-if="activeTab === 4" class="fade-in">
            <h2 class="text-2xl font-bold mb-4 text-indigo-700">Resources & Further Reading</h2>
            
            <div class="grid md:grid-cols-2 gap-4">
              <div>
                <h3 class="text-lg font-semibold mb-2 text-indigo-600">Academic Research</h3>
                <ul class="space-y-2">
                  <li class="p-2 bg-white rounded shadow-sm hover:shadow-md transition-shadow text-sm">
                    <a href="https://doi.org/10.1145/3359246" target="_blank" class="block">
                      <h4 class="font-medium text-indigo-600">Gender Shades (Buolamwini & Gebru, 2018)</h4>
                      <p class="text-gray-600 mt-0.5">Landmark study on racial and gender bias in commercial facial analysis systems.</p>
                    </a>
                  </li>
                  <li class="p-2 bg-white rounded shadow-sm hover:shadow-md transition-shadow text-sm">
                    <a href="https://www.nist.gov/programs-projects/face-recognition-vendor-test-frvt" target="_blank" class="block">
                      <h4 class="font-medium text-indigo-600">NIST Face Recognition Vendor Test (2019)</h4>
                      <p class="text-gray-600 mt-0.5">Comprehensive evaluation of 189 facial recognition algorithms showing demographic differences.</p>
                    </a>
                  </li>
                  <li class="p-2 bg-white rounded shadow-sm hover:shadow-md transition-shadow text-sm">
                    <a href="https://doi.org/10.1145/3442381.3442822" target="_blank" class=" block">
                      <h4 class="font-medium text-indigo-600">The Perpetual Line-Up (Rajput et al., 2020)</h4>
                      <p class="text-gray-600 mt-0.5">Analysis of law enforcement facial recognition networks and privacy implications.</p>
                    </a>
                  </li>
                </ul>
              </div>
              
              <div>
                <h3 class="text-lg font-semibold mb-2 text-indigo-600">Policy Reports</h3>
                <ul class="space-y-2">
                  <li class="p-2 bg-white rounded shadow-sm hover:shadow-md transition-shadow text-sm">
                    <a href="https://www.aclu.org/report/amazon-s-face-recognition-false-match" target="_blank" class="block">
                      <h4 class="font-medium text-indigo-600">ACLU: Amazon's Face Recognition Falsely Matched 28 Members of Congress (2018)</h4>
                      <p class="text-gray-600 mt-0.5">Demonstration of false matches in law enforcement context.</p>
                    </a>
                  </li>
                  <li class="p-2 bg-white rounded shadow-sm hover:shadow-md transition-shadow text-sm">
                    <a href="https://www.eff.org/wp/surveillance-tech" target="_blank" class="block">
                      <h4 class="font-medium text-indigo-600">EFF: Atlas of Surveillance (2021)</h4>
                      <p class="text-gray-600 mt-0.5">Database tracking facial recognition use by law enforcement nationwide.</p>
                    </a>
                  </li>
                  <li class="p-2 bg-white rounded shadow-sm hover:shadow-md transition-shadow text-sm">
                    <a href="https://www.brennancenter.org/our-work/research-reports/how-facial-recognition-threatens-your-civil-liberties" target="_blank" class="block">
                      <h4 class="font-medium text-indigo-600">Brennan Center: How Facial Recognition Threatens Civil Liberties (2020)</h4>
                      <p class="text-gray-600 mt-0.5">Analysis of constitutional and civil rights implications.</p>
                    </a>
                  </li>
                </ul>
              </div>
            </div>
            
            <div class="mt-6">
              <h3 class="text-lg font-semibold mb-2 text-indigo-600">Minnesota-Specific Resources</h3>
              <div class="grid md:grid-cols-3 gap-3">
                <a href="https://www.aclu-mn.org/en/issues/privacy-technology" target="_blank" class="p-2 bg-white rounded shadow-sm hover:shadow-md transition-shadow flex items-center text-sm">
                  <img src="https://www.aclu-mn.org/sites/all/themes/aclu/images/logo.png" alt="ACLU MN" class="h-6 mr-2">
                  <span>ACLU Minnesota Privacy & Technology</span>
                </a>
                <a href="https://www.hf2310.org" target="_blank" class="p-2 bg-white rounded shadow-sm hover:shadow-md transition-shadow flex items-center text-sm">
                  <div class="bg-indigo-100 text-indigo-800 rounded-full w-6 h-6 flex items-center justify-center mr-2">
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                    </svg>
                  </div>
                  <span>MN Facial Recognition Legislation</span>
                </a>
                <a href="https://www.mncourts.gov/mncourtsgov/media/SCAO-Library/Documents/Technology/Facial-Recognition-Report.pdf" target="_blank" class="p-2 bg-white rounded shadow-sm hover:shadow-md transition-shadow flex items-center text-sm">
                  <div class="bg-blue-100 text-blue-800 rounded-full w-6 h-6 flex items-center justify-center mr -2">
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                  </div>
                  <span>MN Judicial Branch Report (2021)</span>
                </a>
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>

    <footer class="bg-gray-800 text-white py-6">
      <div class="container mx-auto px-6">
        <div class="flex flex-col md:flex-row justify-between items-center">
          <div class="mb-3 md:mb-0">
            <h3 class="text-base font-semibold">Facial Recognition Ethics Project</h3>
            <p class="text-gray-400 text-xs">Advocating for responsible technology policy in Minnesota</p>
          </div>
          <div class="flex space-x-4">
            <a href="https://github.com/msentieri/facial-recognition-ethics-project/" class="text-gray-400 hover:text-white transition-colors">
              <span class="sr-only">GitHub</span>
              <svg class="h-5 w-5" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                <path fill-rule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clip-rule="evenodd" />
              </svg>
            </a>
          </div>
        </div>
        <div class="mt-6 pt-4 border-t border-gray-700 text-xs text-gray-400">
          <p>This website is for educational purposes only. The information provided should not be construed as legal advice.</p>
          <p class="mt-1">Last updated: May 2025</p>
        </div>
      </div>
    </footer>
  </div>
</template>

<script>
export default {
  data() {
    return {
      tabs: ['Overview', 'Ethical Issues', 'MN Legislation', 'Try It', 'Resources'],
      activeTab: 0,
      uploadedImages: {
        img1: null,
        img2: null
      },
      isLoading: false,
      result: null,
    };
  },
  methods: {
    handleImageUpload(event, imgKey) {
      const file = event.target.files[0];
      if (file) {
        this.uploadedImages[imgKey] = URL.createObjectURL(file);
      }
    },
    async compareFaces() {
      if (!this.uploadedImages.img1 || !this.uploadedImages.img2) {
        this.result = { error: "Please upload both images first" };
        return;
      }
    
      this.isLoading = true;
      this.result = null;
    
      try {
        
        const credentialsResponse = await fetch('https://facial-recognition-backend-fwzo.onrender.com/api-key');
        if (!credentialsResponse.ok) {
          throw new Error('Failed to get API credentials');
        }
        const { api_key, api_url } = await credentialsResponse.json();
    
        
        const img1Base64 = await this.getBase64(this.uploadedImages.img1);
        const img2Base64 = await this.getBase64(this.uploadedImages.img2);
    
        
        const response = await fetch(api_url, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'subscriptionkey': api_key
          },
          body: JSON.stringify({
            encoded_image1: img1Base64,
            encoded_image2: img2Base64
          })
        });
    
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || 'Comparison failed');
        }
    
        const data = await response.json();
        this.result = {
          verified: data.matchedFaces?.[0]?.matchResult === 1, 
          confidence: data.matchedFaces?.[0]?.quality || 0,
          threshold: 60,
          model: "MXFace API",
          similarity_metric: "Proprietary Algorithm"
        };
      } catch (error) {
        console.error('Error:', error);
        this.result = { error: error.message || "Failed to compare images. Please try again." };
      } finally {
        this.isLoading = false;
      }
    },
    getBase64(imgUrl) {
      return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = 'Anonymous'; // Handle CORS if needed
        img.onload = () => {
          const canvas = document.createElement('canvas');
          const ctx = canvas.getContext('2d');
          canvas.height = img.naturalHeight;
          canvas.width = img.naturalWidth;
          ctx.drawImage(img, 0, 0);
          const dataUrl = canvas.toDataURL('image/jpeg');
          resolve(dataUrl.split(',')[1]); // Extract just the base64 part
        };
        img.onerror = reject;
        img.src = imgUrl;
      });
    }
  }
};
</script>

<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css');

svg {
  height: 1rem !important;
  width: 1rem !important;
}

body {
  font-family: 'Inter', sans-serif;
}

.fade-in {
  animation: fadeIn 0.3s ease-in;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

html {
  scroll-behavior: smooth;
}

::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

::-webkit-scrollbar-track {
  background: #f1f1f1;
}

::-webkit-scrollbar-thumb {
  background: #888;
  border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
  background: #555;
}

.app-container {
  max-width: 900px;
  margin: auto;
  padding: 2rem;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  background: #f5f7fa;
  border-radius: 16px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
}

.title {
  text-align: center;
  font-size: 2.5rem;
  color: #2c3e50;
  margin-bottom: 2rem;
}

.icons-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 1.5rem;
  justify-items: center;
}

.icon-box {
  background: white;
  border-radius: 12px;
  padding: 1rem;
  text-align: center;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
}

.icon-box:hover {
  transform: translateY(-5px);
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
}

.icon {
  font-size: 1.25rem;
  color: #3498db;
  margin-bottom: 0.25rem;
}

.bg-indigo-50 {
  background: linear-gradient(135deg, #eef2ff, #e0e7ff);
}
.bg-yellow-50 {
  background: linear-gradient(135deg, #fef9c3, #fef08a);
}
.bg-blue-50 {
  background: linear-gradient(135deg, #dbeafe, #bfdbfe);
}
.bg-white.rounded-xl {
  background: #ffffff;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.06);
}

h2.text-2xl {
  margin-top: 0.5rem;
}

button.text-sm {
  font-size: 0.875rem;
  border-radius: 0.375rem;
}

.bg-white:hover {
  box-shadow: 0 6px 20px rgba(0, 0, 0, 0.08);
  transform: translateY(-1px);
  transition: all 0.2s ease-in-out;
}
</style>
