use std::sync::Arc;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum GpuError {
    #[error("no suitable hardware GPU adapter found (index={requested_index:?})")]
    NoAdapter { requested_index: Option<usize> },
    #[error("failed to request GPU device: {0}")]
    RequestDevice(#[from] wgpu::RequestDeviceError),
    #[error("GPU buffer map failed: {0}")]
    BufferMap(String),
    #[error("invalid dimensions: {0}")]
    InvalidDimensions(String),
}

#[derive(Debug, Clone, Default)]
pub struct GpuOptions {
    /// Explicit 0-based hardware adapter index (overrides `COGZILLA_GPU_INDEX`).
    pub adapter_index: Option<usize>,
    /// Allow software fallback adapters (e.g. llvmpipe) if no hardware GPU is found.
    pub allow_software: bool,
}

#[derive(Debug, Clone)]
pub struct GpuAdapterInfo {
    pub index: usize,
    pub name: String,
    pub backend: wgpu::Backend,
    pub device_type: wgpu::DeviceType,
    pub driver: String,
    pub driver_info: String,
    /// True on Apple Silicon / integrated GPUs where CPU and GPU share physical memory.
    pub unified_memory: bool,
}

impl GpuAdapterInfo {
    fn from_wgpu(index: usize, info: &wgpu::AdapterInfo) -> Self {
        let unified_memory = info.device_type == wgpu::DeviceType::IntegratedGpu
            && info.backend == wgpu::Backend::Metal;
        Self {
            index,
            name: info.name.clone(),
            backend: info.backend,
            device_type: info.device_type,
            driver: info.driver.clone(),
            driver_info: info.driver_info.clone(),
            unified_memory,
        }
    }
}

/// Shared `wgpu` device, queue, and adapter metadata.
pub struct GpuContext {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub info: GpuAdapterInfo,
}

impl GpuContext {
    /// Enumerate available GPU adapters (hardware adapters first, excluding software CPU renderers
    /// unless `allow_software` is true).
    pub fn list_adapters(allow_software: bool) -> Vec<GpuAdapterInfo> {
        let instance = Self::create_instance();
        Self::enumerate_hardware(&instance, allow_software)
            .into_iter()
            .enumerate()
            .map(|(i, a)| GpuAdapterInfo::from_wgpu(i, &a.get_info()))
            .collect()
    }

    fn create_instance() -> wgpu::Instance {
        let backends = wgpu::Backends::METAL | wgpu::Backends::VULKAN | wgpu::Backends::DX12;
        wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        })
    }

    fn enumerate_hardware(instance: &wgpu::Instance, allow_software: bool) -> Vec<wgpu::Adapter> {
        let mut adapters: Vec<wgpu::Adapter> = instance
            .enumerate_adapters(wgpu::Backends::METAL | wgpu::Backends::VULKAN | wgpu::Backends::DX12)
            .into_iter()
            .filter(|a| {
                let dt = a.get_info().device_type;
                allow_software || dt != wgpu::DeviceType::Cpu
            })
            .collect();
        // Prefer DiscreteGpu and IntegratedGpu ahead of VirtualGpu / Other / Cpu.
        adapters.sort_by_key(|a| match a.get_info().device_type {
            wgpu::DeviceType::DiscreteGpu => 0,
            wgpu::DeviceType::IntegratedGpu => 1,
            wgpu::DeviceType::VirtualGpu => 2,
            wgpu::DeviceType::Other => 3,
            wgpu::DeviceType::Cpu => 4,
        });
        adapters
    }

    /// Initialize a `GpuContext` using `options` (or `COGZILLA_GPU_INDEX` env var if set).
    pub fn new(options: GpuOptions) -> Result<Self, GpuError> {
        let requested_index = options.adapter_index.or_else(|| {
            std::env::var("COGZILLA_GPU_INDEX")
                .ok()
                .and_then(|s| s.trim().parse::<usize>().ok())
        });

        let instance = Self::create_instance();
        let adapters = Self::enumerate_hardware(&instance, options.allow_software);
        let chosen_idx = requested_index.unwrap_or(0);
        let adapter = adapters
            .into_iter()
            .nth(chosen_idx)
            .ok_or(GpuError::NoAdapter {
                requested_index: Some(chosen_idx),
            })?;

        let info = GpuAdapterInfo::from_wgpu(chosen_idx, &adapter.get_info());
        let adapter_limits = adapter.limits();

        // Request generous storage buffer limits (up to 256 MiB per binding if supported)
        // so 4096x4096+ f32 slabs and outputs fit comfortably.
        let mut limits = wgpu::Limits::default();
        limits.max_storage_buffer_binding_size = adapter_limits
            .max_storage_buffer_binding_size
            .min(512 * 1024 * 1024);
        limits.max_buffer_size = adapter_limits.max_buffer_size.min(512 * 1024 * 1024);

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("cogzilla-gpu-device"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))?;

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            info,
        })
    }

    /// Convenience constructor using default options (`COGZILLA_GPU_INDEX` if set, else adapter 0).
    pub fn new_default() -> Result<Self, GpuError> {
        Self::new(GpuOptions::default())
    }
}
