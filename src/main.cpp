#include <QApplication>
#include <QCoreApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QIcon>
#include <memory>
#include "profilemanager.h"
#include "settingsmanager.h"
#include "llmprovider.h"
#include "orchestrator.h"
#include "memorystore.h"
#include "idlescheduler.h"
#include "whyengine.h"
#include "traitextractor.h"
#include "evalsuite.h"
#include "schemamigrator.h"
#include "modelgeneration.h"
#include "profilehealthmanager.h"
#include "datalifecyclemanager.h"
#include "websearchengine.h"
#include "researchstore.h"
#include "researchengine.h"
#include "thoughtengine.h"
#include "newsengine.h"
#include "reminderengine.h"
#include "goaltracker.h"
#include "sentimenttracker.h"
#include "learningengine.h"
#include "clipboardhelper.h"
#include "embeddingmanager.h"

#ifdef DATAFORM_TRAINING_ENABLED
#include "tokenizer.h"
#include "trainingdatagenerator.h"
#include "orttrainingmanager.h"
#include "adaptermanager.h"
#include "reflectionengine.h"
#include "populationstate.h"
#include "weightmerger.h"
#include "evolutionengine.h"
#include "lineagetracker.h"
#include "ortinferencemanager.h"
#endif

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    QCoreApplication::setOrganizationName("GHI");
    QCoreApplication::setOrganizationDomain("globalhumaninitiative.org");
    QCoreApplication::setApplicationName("DATAFORM");
    app.setWindowIcon(QIcon(QCoreApplication::applicationDirPath() + "/dataform.ico"));

    // --- Create components in dependency order ---

    // 1. Profile manager creates directory tree
    ProfileManager profileManager;

    // 2. Settings manager loads from profile path (JSON migration runs in loadSettings)
    SettingsManager settingsManager(profileManager.profilePath());

    // 3. Model generation manager (Phase 4)
    ModelGenerationManager modelGenManager(profileManager.profilePath());

    // 4. LLM provider for inference
    LLMProviderManager llmProvider;

    // 5. Memory store for episodic + trait persistence (DB migration runs in initialize)
    MemoryStore memoryStore(profileManager.profilePath());

    // 5. Orchestrator implements the cognitive loop
    Orchestrator orchestrator;

    // 6. Idle scheduler for background learning windows
    IdleScheduler idleScheduler;

    // 7. Why Engine - curiosity controller (Phase 1)
    WhyEngine whyEngine;

    // 8. Trait Extractor - LLM-based trait extraction (Phase 1)
    TraitExtractor traitExtractor;

    // 9. Evaluation Suite - identity alignment testing (Phase 1)
    EvalSuite evalSuite;

    // 10. Profile Health Manager (Phase 4)
    ProfileHealthManager profileHealthManager;

    // 11. Data Lifecycle Manager (Phase 4)
    DataLifecycleManager dataLifecycleManager;

    // 12. Phase 5: Autonomous Research System
    WebSearchEngine webSearchEngine;
    ResearchStore researchStore;
    ResearchEngine researchEngine;

    // 13. Proactive Dialog: ThoughtEngine
    ThoughtEngine thoughtEngine;

    // 14. News Engine: idle-time news headlines
    NewsEngine newsEngine;

    // 15. Agentic Features: Reminders, Goals, Sentiment, Learning
    ReminderEngine reminderEngine;
    GoalTracker goalTracker;
    SentimentTracker sentimentTracker;
    LearningEngine learningEngine;

    // 16. Clipboard helper for image paste support
    ClipboardHelper clipboardHelper;

    // 17. Embedding Manager (Phase 7: Semantic Memory)
    EmbeddingManager embeddingManager;

#ifdef DATAFORM_TRAINING_ENABLED
    // 12. Tokenizer for ORT training (Phase 2)
    Tokenizer tokenizer;

    // 11. Training data generator (Phase 2)
    TrainingDataGenerator trainingDataGenerator;

    // 12. ORT Training Manager (Phase 2)
    OrtTrainingManager ortTrainingManager;

    // 13. Adapter Manager (Phase 2)
    AdapterManager adapterManager;

    // 14. Reflection Engine - idle-time training orchestrator (Phase 2)
    ReflectionEngine reflectionEngine;

    // 15. Weight Merger utility (Phase 3)
    WeightMerger weightMerger;

    // 16. Evolution Engine - Darwinian evolution orchestrator (Phase 3)
    EvolutionEngine evolutionEngine;

    // 17. Lineage Tracker (Phase 4)
    LineageTracker lineageTracker;

    // 18. ORT Inference Manager - local model inference (Phase 6)
    OrtInferenceManager ortInferenceManager;
#endif

    // --- Wire dependencies ---

    orchestrator.setLLMProvider(&llmProvider);
    orchestrator.setMemoryStore(&memoryStore);
    orchestrator.setSettingsManager(&settingsManager);
    orchestrator.setWhyEngine(&whyEngine);
    orchestrator.setTraitExtractor(&traitExtractor);

    whyEngine.setMemoryStore(&memoryStore);
    traitExtractor.setLLMProvider(&llmProvider);
    traitExtractor.setMemoryStore(&memoryStore);
    evalSuite.setMemoryStore(&memoryStore);
    idleScheduler.setMemoryStore(&memoryStore);

    // Wire Phase 4 dependencies
    profileHealthManager.setProfileManager(&profileManager);
    profileHealthManager.setMemoryStore(&memoryStore);
    profileHealthManager.setSettingsManager(&settingsManager);

    dataLifecycleManager.setMemoryStore(&memoryStore);
    dataLifecycleManager.setProfileManager(&profileManager);
    dataLifecycleManager.setSettingsManager(&settingsManager);

    // Wire Phase 5 dependencies
    researchStore.setMemoryStore(&memoryStore);
    researchEngine.setWebSearchEngine(&webSearchEngine);
    researchEngine.setMemoryStore(&memoryStore);
    researchEngine.setResearchStore(&researchStore);
    researchEngine.setLLMProvider(&llmProvider);
    researchEngine.setWhyEngine(&whyEngine);
    researchEngine.setSettingsManager(&settingsManager);
    orchestrator.setResearchEngine(&researchEngine);

    // Wire ThoughtEngine dependencies
    thoughtEngine.setMemoryStore(&memoryStore);
    thoughtEngine.setResearchStore(&researchStore);
    thoughtEngine.setResearchEngine(&researchEngine);
    thoughtEngine.setWhyEngine(&whyEngine);
    thoughtEngine.setLLMProvider(&llmProvider);
    thoughtEngine.setSettingsManager(&settingsManager);
    orchestrator.setThoughtEngine(&thoughtEngine);

    // Wire NewsEngine dependencies
    newsEngine.setMemoryStore(&memoryStore);
    newsEngine.setLLMProvider(&llmProvider);
    newsEngine.setThoughtEngine(&thoughtEngine);
    newsEngine.setSettingsManager(&settingsManager);

    // Wire Agentic Features
    reminderEngine.setMemoryStore(&memoryStore);
    reminderEngine.setThoughtEngine(&thoughtEngine);
    reminderEngine.setSettingsManager(&settingsManager);
    orchestrator.setReminderEngine(&reminderEngine);

    goalTracker.setMemoryStore(&memoryStore);
    goalTracker.setLLMProvider(&llmProvider);
    goalTracker.setThoughtEngine(&thoughtEngine);
    goalTracker.setSettingsManager(&settingsManager);
    orchestrator.setGoalTracker(&goalTracker);

    sentimentTracker.setMemoryStore(&memoryStore);
    sentimentTracker.setLLMProvider(&llmProvider);
    sentimentTracker.setThoughtEngine(&thoughtEngine);
    sentimentTracker.setSettingsManager(&settingsManager);
    orchestrator.setSentimentTracker(&sentimentTracker);

    learningEngine.setMemoryStore(&memoryStore);
    learningEngine.setLLMProvider(&llmProvider);
    learningEngine.setThoughtEngine(&thoughtEngine);
    learningEngine.setResearchEngine(&researchEngine);
    learningEngine.setSettingsManager(&settingsManager);
    orchestrator.setLearningEngine(&learningEngine);
    orchestrator.setProfileManager(&profileManager);

    // Wire Phase 7: Embedding Manager (Semantic Memory)
    embeddingManager.setMemoryStore(&memoryStore);
    embeddingManager.setSettingsManager(&settingsManager);
    orchestrator.setEmbeddingManager(&embeddingManager);

#ifdef DATAFORM_TRAINING_ENABLED
    // Wire Phase 6: ORT Inference Manager
    ortInferenceManager.setTokenizer(&tokenizer);
    llmProvider.setOrtInferenceManager(&ortInferenceManager);

    // Wire Phase 2 dependencies
    trainingDataGenerator.setMemoryStore(&memoryStore);
    trainingDataGenerator.setTokenizer(&tokenizer);
    trainingDataGenerator.setResearchStore(&researchStore);

    adapterManager.setProfileManager(&profileManager);
    adapterManager.setEvalSuite(&evalSuite);

    reflectionEngine.setMemoryStore(&memoryStore);
    reflectionEngine.setTokenizer(&tokenizer);
    reflectionEngine.setTrainingDataGenerator(&trainingDataGenerator);
    reflectionEngine.setOrtTrainingManager(&ortTrainingManager);
    reflectionEngine.setAdapterManager(&adapterManager);
    reflectionEngine.setEvalSuite(&evalSuite);
    reflectionEngine.setIdleScheduler(&idleScheduler);
    reflectionEngine.setProfileManager(&profileManager);
    reflectionEngine.setSettingsManager(&settingsManager);
    reflectionEngine.setLLMProvider(&llmProvider);
    reflectionEngine.setOrtInferenceManager(&ortInferenceManager);

    // Wire Phase 3 (Evolution Engine) dependencies
    evolutionEngine.setMemoryStore(&memoryStore);
    evolutionEngine.setTokenizer(&tokenizer);
    evolutionEngine.setTrainingDataGenerator(&trainingDataGenerator);
    evolutionEngine.setOrtTrainingManager(&ortTrainingManager);
    evolutionEngine.setAdapterManager(&adapterManager);
    evolutionEngine.setEvalSuite(&evalSuite);
    evolutionEngine.setIdleScheduler(&idleScheduler);
    evolutionEngine.setProfileManager(&profileManager);
    evolutionEngine.setSettingsManager(&settingsManager);
    evolutionEngine.setLLMProvider(&llmProvider);
    evolutionEngine.setOrtInferenceManager(&ortInferenceManager);
    evolutionEngine.setWeightMerger(&weightMerger);

    // Wire Phase 4 lineage tracker
    lineageTracker.setProfileManager(&profileManager);
    lineageTracker.setAdapterManager(&adapterManager);
    lineageTracker.setMemoryStore(&memoryStore);
    evolutionEngine.setLineageTracker(&lineageTracker);

    // Connect idle scheduler to evolution engine (Phase 3 replaces Phase 2 as idle consumer)
    QObject::connect(&idleScheduler, &IdleScheduler::idleWindowOpened,
                     &evolutionEngine, &EvolutionEngine::onIdleWindowOpened);
    QObject::connect(&idleScheduler, &IdleScheduler::idleWindowClosed,
                     &evolutionEngine, &EvolutionEngine::onIdleWindowClosed);
#endif

    // --- Load settings and configure ---

    settingsManager.loadSettings();
    llmProvider.setCurrentProvider(settingsManager.provider());
    llmProvider.setCurrentModel(settingsManager.model());
    llmProvider.setOpenAIKey(settingsManager.openAIKey());
    llmProvider.setAnthropicKey(settingsManager.anthropicKey());
    llmProvider.setLmStudioUrl(settingsManager.lmStudioUrl());
    llmProvider.setOllamaUrl(settingsManager.ollamaUrl());

    // Sync settings changes to LLM provider + auto-save LLM config
    QObject::connect(&settingsManager, &SettingsManager::providerChanged,
                     &llmProvider, [&]() {
        llmProvider.setCurrentProvider(settingsManager.provider());
        settingsManager.saveSettings();
    });
    QObject::connect(&settingsManager, &SettingsManager::modelChanged,
                     &llmProvider, [&]() {
        llmProvider.setCurrentModel(settingsManager.model());
        settingsManager.saveSettings();
    });
    QObject::connect(&settingsManager, &SettingsManager::openAIKeyChanged,
                     &llmProvider, [&]() {
        llmProvider.setOpenAIKey(settingsManager.openAIKey());
        settingsManager.saveSettings();
    });
    QObject::connect(&settingsManager, &SettingsManager::anthropicKeyChanged,
                     &llmProvider, [&]() {
        llmProvider.setAnthropicKey(settingsManager.anthropicKey());
        settingsManager.saveSettings();
    });
    QObject::connect(&settingsManager, &SettingsManager::lmStudioUrlChanged,
                     &llmProvider, [&]() {
        llmProvider.setLmStudioUrl(settingsManager.lmStudioUrl());
        settingsManager.saveSettings();
    });
    QObject::connect(&settingsManager, &SettingsManager::ollamaUrlChanged,
                     &llmProvider, [&]() {
        llmProvider.setOllamaUrl(settingsManager.ollamaUrl());
        settingsManager.saveSettings();
    });

    // Auto-refresh models on startup: fetch real model list from Ollama/LM Studio,
    // then re-apply the saved model and test connection.
    // Capture saved model now — QML ComboBox may overwrite settingsManager.model
    // before the async refresh completes (because the saved model isn't in the
    // hardcoded default list, so the ComboBox resets to index 0).
    QString startupSavedModel = settingsManager.model();
    QString startupProvider = settingsManager.provider();

    // One-shot connection: restore saved model after first refresh, then disconnect
    auto startupConn = std::make_shared<QMetaObject::Connection>();
    *startupConn = QObject::connect(&llmProvider, &LLMProviderManager::modelsRefreshed,
                     &llmProvider, [&, startupSavedModel, startupConn](bool success, const QString &) {
        QObject::disconnect(*startupConn);
        if (success && !startupSavedModel.isEmpty()) {
            if (llmProvider.availableModels().contains(startupSavedModel)) {
                llmProvider.setCurrentModel(startupSavedModel);
                settingsManager.setModel(startupSavedModel);
            }
        }
        llmProvider.testConnection();
    });

    if (startupProvider == "Ollama" || startupProvider == "LM Studio") {
        llmProvider.refreshModels();
    } else if (startupProvider == "Local" || startupProvider == "OpenAI" || startupProvider == "Anthropic") {
        llmProvider.testConnection();
    }

    // Sync settings changes to idle scheduler
    QObject::connect(&settingsManager, &SettingsManager::idleLearningEnabledChanged,
                     &idleScheduler, [&]() {
        idleScheduler.setEnabled(settingsManager.idleLearningEnabled());
    });
    QObject::connect(&settingsManager, &SettingsManager::computeBudgetPercentChanged,
                     &idleScheduler, [&]() {
        idleScheduler.setComputeBudgetPercent(settingsManager.computeBudgetPercent());
    });

    // --- Initialize memory store (runs DB migrations) ---

    memoryStore.setSettingsManager(&settingsManager);
    memoryStore.initialize();

    // Re-encrypt databases when encryption mode changes
    QObject::connect(&settingsManager, &SettingsManager::encryptionModeChanged,
                     &memoryStore, [&]() {
        qDebug() << "Encryption mode changed to:" << settingsManager.encryptionMode()
                 << "- re-encrypting databases...";
        memoryStore.flush();
    });

    // Phase 5: Initialize research store (needs episodic DB from memoryStore)
    researchStore.initialize();

    // Initialize ThoughtEngine (needs episodic DB from memoryStore)
    thoughtEngine.initialize();

    // Phase 7: Load embeddings into memory and check model availability
    embeddingManager.loadEmbeddingsFromDb();
    embeddingManager.checkModelAvailability();

    // --- Phase 4: Load model generations ---
    modelGenManager.loadGenerations();

    // --- Phase 4: Run startup health check ---
    profileHealthManager.runStartupCheck();

#ifdef DATAFORM_TRAINING_ENABLED
    // Configure tokenizer from current generation
    tokenizer.configure(modelGenManager.currentGeneration());

    // Load tokenizer from training artifacts
    QString tokenizerVocab = profileManager.profilePath() + "/training_artifacts/vocab.json";
    QString tokenizerMerges = profileManager.profilePath() + "/training_artifacts/merges.txt";
    if (QFile::exists(tokenizerVocab) && QFile::exists(tokenizerMerges)) {
        tokenizer.loadFromFiles(tokenizerVocab, tokenizerMerges);
    }

    // Load existing adapter versions
    adapterManager.loadVersions();

    // Load evolution state (resume interrupted cycles)
    evolutionEngine.loadPopulationState();

    // Build lineage from adapter history
    lineageTracker.buildLineage();

    // Phase 6: Load active adapter model for local inference at startup
    {
        AdapterMetadata active = adapterManager.activeAdapterMetadata();
        if (!active.exportedModelPath.isEmpty() && QFile::exists(active.exportedModelPath)) {
            ortInferenceManager.loadModel(active.exportedModelPath);
            qDebug() << "Loaded active adapter for local inference:" << active.exportedModelPath;
        } else {
            // Fall back to eval_model.onnx from training artifacts (base model)
            QString evalModel = profileManager.profilePath() + "/training_artifacts/eval_model.onnx";
            if (QFile::exists(evalModel)) {
                ortInferenceManager.loadModel(evalModel);
                qDebug() << "Loaded base eval model for local inference:" << evalModel;
            }
        }
    }
#endif

    // --- Phase 4: Initial disk usage scan ---
    dataLifecycleManager.updateDiskUsage();

    // --- Start idle scheduler ---

    idleScheduler.setComputeBudgetPercent(settingsManager.computeBudgetPercent());
    idleScheduler.setEnabled(settingsManager.idleLearningEnabled());

    // Connect idle scheduler to lifecycle sweep (Phase 4)
    QObject::connect(&idleScheduler, &IdleScheduler::idleWindowOpened,
                     &dataLifecycleManager, &DataLifecycleManager::runLifecycleSweep);

    // Connect idle scheduler to research engine (Phase 5)
    QObject::connect(&idleScheduler, &IdleScheduler::idleWindowOpened,
                     &researchEngine, &ResearchEngine::onIdleWindowOpened);
    QObject::connect(&idleScheduler, &IdleScheduler::idleWindowClosed,
                     &researchEngine, &ResearchEngine::onIdleWindowClosed);

    // Connect idle scheduler to ThoughtEngine (Proactive Dialog)
    QObject::connect(&idleScheduler, &IdleScheduler::idleWindowOpened,
                     &thoughtEngine, &ThoughtEngine::onIdleWindowOpened);
    QObject::connect(&idleScheduler, &IdleScheduler::idleWindowClosed,
                     &thoughtEngine, &ThoughtEngine::onIdleWindowClosed);

    // Connect research events to ThoughtEngine
    QObject::connect(&researchEngine, &ResearchEngine::researchCycleComplete,
                     &thoughtEngine, &ThoughtEngine::onResearchCycleComplete);

    // Connect idle scheduler to NewsEngine
    QObject::connect(&idleScheduler, &IdleScheduler::idleWindowOpened,
                     &newsEngine, &NewsEngine::onIdleWindowOpened);
    QObject::connect(&idleScheduler, &IdleScheduler::idleWindowClosed,
                     &newsEngine, &NewsEngine::onIdleWindowClosed);

    // Connect news events to ThoughtEngine
    QObject::connect(&newsEngine, &NewsEngine::newsCycleComplete,
                     &thoughtEngine, &ThoughtEngine::onNewsCycleComplete);

    // Start ReminderEngine polling timer
    reminderEngine.start();

    // Connect idle scheduler to GoalTracker
    QObject::connect(&idleScheduler, &IdleScheduler::idleWindowOpened,
                     &goalTracker, &GoalTracker::onIdleWindowOpened);
    QObject::connect(&idleScheduler, &IdleScheduler::idleWindowClosed,
                     &goalTracker, &GoalTracker::onIdleWindowClosed);

    // Connect idle scheduler to TraitExtractor (conversation scanning)
    QObject::connect(&idleScheduler, &IdleScheduler::idleWindowOpened,
                     &traitExtractor, &TraitExtractor::onIdleWindowOpened);
    QObject::connect(&idleScheduler, &IdleScheduler::idleWindowClosed,
                     &traitExtractor, &TraitExtractor::onIdleWindowClosed);

    // Connect episode storage to SentimentTracker
    QObject::connect(&orchestrator, &Orchestrator::episodeStored,
                     &sentimentTracker, &SentimentTracker::onEpisodeStored);
    QObject::connect(&idleScheduler, &IdleScheduler::idleWindowOpened,
                     &sentimentTracker, &SentimentTracker::onIdleWindowOpened);

    // Connect idle scheduler to LearningEngine
    QObject::connect(&idleScheduler, &IdleScheduler::idleWindowOpened,
                     &learningEngine, &LearningEngine::onIdleWindowOpened);
    QObject::connect(&idleScheduler, &IdleScheduler::idleWindowClosed,
                     &learningEngine, &LearningEngine::onIdleWindowClosed);

    // Connect idle scheduler to EmbeddingManager (Phase 7: lowest priority)
    QObject::connect(&idleScheduler, &IdleScheduler::idleWindowOpened,
                     &embeddingManager, &EmbeddingManager::onIdleWindowOpened);
    QObject::connect(&idleScheduler, &IdleScheduler::idleWindowClosed,
                     &embeddingManager, &EmbeddingManager::onIdleWindowClosed);
    QObject::connect(&orchestrator, &Orchestrator::episodeStored,
                     &embeddingManager, &EmbeddingManager::onEpisodeInserted);

#ifdef DATAFORM_TRAINING_ENABLED
    // Connect training events to ThoughtEngine
    QObject::connect(&evolutionEngine, &EvolutionEngine::evolutionCycleComplete,
                     &thoughtEngine, &ThoughtEngine::onEvolutionCycleComplete);
    QObject::connect(&reflectionEngine, &ReflectionEngine::reflectionComplete,
                     &thoughtEngine, &ThoughtEngine::onReflectionComplete);
#endif

    idleScheduler.start();

    // --- Clean shutdown ---

    QObject::connect(&app, &QCoreApplication::aboutToQuit, [&]() {
        idleScheduler.stop();
        reminderEngine.stop();
        goalTracker.onIdleWindowClosed();
        learningEngine.onIdleWindowClosed();
        newsEngine.onIdleWindowClosed();
        researchEngine.pauseResearch();
        embeddingManager.onIdleWindowClosed();
        dataLifecycleManager.runLifecycleSweep();
#ifdef DATAFORM_TRAINING_ENABLED
        evolutionEngine.pauseEvolution();
        if (ortTrainingManager.isTraining()) {
            ortTrainingManager.requestPause();
            QThread::msleep(2000);
        }
#endif
        memoryStore.close();
    });

    // --- QML Engine ---

    QQmlApplicationEngine engine;

    engine.rootContext()->setContextProperty("profileManager", &profileManager);
    engine.rootContext()->setContextProperty("settingsManager", &settingsManager);
    engine.rootContext()->setContextProperty("llmProvider", &llmProvider);
    engine.rootContext()->setContextProperty("orchestrator", &orchestrator);
    engine.rootContext()->setContextProperty("memoryStore", &memoryStore);
    engine.rootContext()->setContextProperty("idleScheduler", &idleScheduler);
    engine.rootContext()->setContextProperty("whyEngine", &whyEngine);
    engine.rootContext()->setContextProperty("traitExtractor", &traitExtractor);
    engine.rootContext()->setContextProperty("evalSuite", &evalSuite);
    engine.rootContext()->setContextProperty("modelGenManager", &modelGenManager);
    engine.rootContext()->setContextProperty("profileHealthManager", &profileHealthManager);
    engine.rootContext()->setContextProperty("dataLifecycleManager", &dataLifecycleManager);
    engine.rootContext()->setContextProperty("webSearchEngine", &webSearchEngine);
    engine.rootContext()->setContextProperty("researchStore", &researchStore);
    engine.rootContext()->setContextProperty("researchEngine", &researchEngine);
    engine.rootContext()->setContextProperty("thoughtEngine", &thoughtEngine);
    engine.rootContext()->setContextProperty("newsEngine", &newsEngine);
    engine.rootContext()->setContextProperty("reminderEngine", &reminderEngine);
    engine.rootContext()->setContextProperty("goalTracker", &goalTracker);
    engine.rootContext()->setContextProperty("sentimentTracker", &sentimentTracker);
    engine.rootContext()->setContextProperty("learningEngine", &learningEngine);
    engine.rootContext()->setContextProperty("clipboardHelper", &clipboardHelper);
    engine.rootContext()->setContextProperty("embeddingManager", &embeddingManager);

#ifdef DATAFORM_TRAINING_ENABLED
    engine.rootContext()->setContextProperty("tokenizer", &tokenizer);
    engine.rootContext()->setContextProperty("trainingDataGenerator", &trainingDataGenerator);
    engine.rootContext()->setContextProperty("ortTrainingManager", &ortTrainingManager);
    engine.rootContext()->setContextProperty("adapterManager", &adapterManager);
    engine.rootContext()->setContextProperty("reflectionEngine", &reflectionEngine);
    engine.rootContext()->setContextProperty("weightMerger", &weightMerger);
    engine.rootContext()->setContextProperty("evolutionEngine", &evolutionEngine);
    engine.rootContext()->setContextProperty("lineageTracker", &lineageTracker);
    engine.rootContext()->setContextProperty("ortInferenceManager", &ortInferenceManager);
#endif

    const QUrl url(u"qrc:/Dataform/qml/Main.qml"_qs);

    QObject::connect(
        &engine,
        &QQmlApplicationEngine::objectCreationFailed,
        &app,
        [&]() {
            qCritical() << "QML object creation FAILED";
            QCoreApplication::exit(-1);
        },
        Qt::QueuedConnection
    );

    engine.load(url);

    return app.exec();
}
