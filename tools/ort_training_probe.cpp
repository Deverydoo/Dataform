// Verifies that the locally built onnxruntime.dll really carries the on-device
// training API, and that its function pointers are live rather than merely
// present. GetTrainingApi returning null is the failure mode that would
// otherwise only surface during an idle-time training run.
#include <cstdio>
#include "onnxruntime_training_cxx_api.h"

int main() {
    const OrtApi *api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!api) {
        std::printf("FAIL: GetApi(%d) returned null\n", ORT_API_VERSION);
        return 1;
    }
    std::printf("OK   base API requested v%d, runtime version string = %s\n",
                ORT_API_VERSION, OrtGetApiBase()->GetVersionString());

    // Muninn is compiled against ORT 1.23, so it asks for API 23. Prove this
    // runtime satisfies that too, since one DLL has to serve both.
    if (!OrtGetApiBase()->GetApi(23)) {
        std::printf("FAIL: GetApi(23) returned null - Muninn's neural VAD would break\n");
        return 2;
    }
    std::printf("OK   GetApi(23) satisfied - Muninn compatible\n");

    const OrtTrainingApi *training = api->GetTrainingApi(ORT_API_VERSION);
    if (!training) {
        std::printf("FAIL: GetTrainingApi(%d) returned null - this DLL has no training support\n",
                    ORT_API_VERSION);
        return 3;
    }
    std::printf("OK   training API struct present\n");

    // Call through a real training function pointer. A missing checkpoint must
    // come back as a clean OrtStatus; a crash here would mean the struct is
    // populated with garbage.
    OrtCheckpointState *state = nullptr;
    OrtStatus *status = training->LoadCheckpoint(ORT_TSTR("no_such_checkpoint_on_disk"), &state);
    if (status) {
        std::printf("OK   LoadCheckpoint rejected a bogus path cleanly: %s\n",
                    api->GetErrorMessage(status));
        api->ReleaseStatus(status);
    } else {
        std::printf("WARN LoadCheckpoint unexpectedly succeeded on a bogus path\n");
        if (state)
            training->ReleaseCheckpointState(state);
    }

    std::printf("\nRESULT: training API is functional on this build.\n");
    return 0;
}
