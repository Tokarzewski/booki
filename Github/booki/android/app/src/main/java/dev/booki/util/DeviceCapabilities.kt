package dev.booki.util

import android.app.ActivityManager
import android.content.Context
import android.os.Build

/** Hardware facts the UI uses to gate premium engines. */
data class DeviceCapabilities(
    val ramMb: Long,
    val abis: List<String>,
    val sdkInt: Int,
) {
    val isArm64: Boolean get() = "arm64-v8a" in abis

    companion object {
        fun of(context: Context): DeviceCapabilities {
            val mi = ActivityManager.MemoryInfo()
            (context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager).getMemoryInfo(mi)
            return DeviceCapabilities(
                ramMb = mi.totalMem / 1_048_576L,
                abis = Build.SUPPORTED_ABIS.toList(),
                sdkInt = Build.VERSION.SDK_INT,
            )
        }
    }
}
