use llvm_sys::core::LLVMAddFunction;
use llvm_sys::core::LLVMAddGlobal;
use llvm_sys::core::LLVMAppendBasicBlock;
use llvm_sys::core::LLVMArrayType;
use llvm_sys::core::LLVMBuildAdd;
use llvm_sys::core::LLVMBuildAlloca;
use llvm_sys::core::LLVMBuildAnd;
use llvm_sys::core::LLVMBuildBr;
use llvm_sys::core::LLVMBuildCall;
use llvm_sys::core::LLVMBuildCondBr;
use llvm_sys::core::LLVMBuildICmp;
use llvm_sys::core::LLVMBuildLoad;
use llvm_sys::core::LLVMBuildMul;
use llvm_sys::core::LLVMBuildNot;
use llvm_sys::core::LLVMBuildOr;
use llvm_sys::core::LLVMBuildPointerCast;
use llvm_sys::core::LLVMBuildRet;
use llvm_sys::core::LLVMBuildSDiv;
use llvm_sys::core::LLVMBuildStore;
use llvm_sys::core::LLVMBuildSub;
use llvm_sys::core::LLVMConstInt;
use llvm_sys::core::LLVMConstString;
use llvm_sys::core::LLVMContextCreate;
use llvm_sys::core::LLVMContextDispose;
use llvm_sys::core::LLVMCountParams;
use llvm_sys::core::LLVMCreateBuilderInContext;
use llvm_sys::core::LLVMDisposeBuilder;
use llvm_sys::core::LLVMDisposeModule;
use llvm_sys::core::LLVMFunctionType;
use llvm_sys::core::LLVMGetParam;
use llvm_sys::core::LLVMIsMultithreaded;
use llvm_sys::core::LLVMModuleCreateWithNameInContext;
use llvm_sys::core::LLVMPositionBuilderAtEnd;
use llvm_sys::core::LLVMPrintModuleToFile;
use llvm_sys::core::LLVMSetGlobalConstant;
use llvm_sys::core::LLVMSetInitializer;
use llvm_sys::core::{
    LLVMInt1TypeInContext, LLVMInt32TypeInContext, LLVMInt8TypeInContext, LLVMPointerType,
};
use llvm_sys::prelude::*;
use llvm_sys::LLVMBuilder;
use llvm_sys::LLVMModule;
use std::ffi::CString;
use std::os::raw::c_uint;
use std::os::raw::c_ulonglong;
use std::ptr::null_mut;
use std::rc::Rc;

const LLVM_FALSE: LLVMBool = 0;
const LLVM_TRUE: LLVMBool = 1;

/// A module is the highest-level unit of compilation in LLVM.
pub struct Module {
    module: *mut LLVMModule,
    strings: Vec<CString>,
    empty_name: *const i8,
    blocks: Vec<LLVMBasicBlockRef>,
    values: Vec<LLVMValueRef>,
    functions: Vec<(LLVMValueRef, bool)>,
}

struct Builder {
    builder: *mut LLVMBuilder,
}

impl Builder {
    /// Create a new Builder in our LLVM context context.
    fn for_block(bb: LLVMBasicBlockRef) -> Self {
        unsafe {
            let builder = CONTEXT.with(|c| {
                let builder = LLVMCreateBuilderInContext(c.context);
                LLVMPositionBuilderAtEnd(builder, bb);
                builder
            });
            Builder { builder }
        }
    }
}

impl Drop for Builder {
    fn drop(&mut self) {
        unsafe {
            LLVMDisposeBuilder(self.builder);
        }
    }
}

struct Context {
    context: LLVMContextRef,
}

impl Context {
    fn new() -> Self {
        Self {
            context: unsafe { LLVMContextCreate() },
        }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe { LLVMContextDispose(self.context) }
    }
}

thread_local!(static CONTEXT: Context = Context::new());

impl Module {
    pub fn new(module_name: &str) -> Self {
        assert!(unsafe { LLVMIsMultithreaded() } != 0);
        let c_module_name = CString::new(module_name).unwrap();
        let module_name_char_ptr = c_module_name.to_bytes_with_nul().as_ptr() as *const _;

        let llvm_module = unsafe {
            CONTEXT.with(|c| LLVMModuleCreateWithNameInContext(module_name_char_ptr, c.context))
        };
        let empty_name = CString::new("").unwrap();
        let empty_name_ptr = empty_name.to_bytes_with_nul().as_ptr() as *const _;
        let module = Module {
            module: llvm_module,
            strings: vec![c_module_name, empty_name],
            empty_name: empty_name_ptr,
            blocks: vec![],
            values: vec![],
            functions: vec![],
        };
        module
    }

    pub fn write(&mut self, filename: &str) {
        unsafe {
            LLVMPrintModuleToFile(self.module, self.new_string_ptr(filename), null_mut());
        }
    }

    pub fn static_bool(&mut self, val: bool) -> LLVMValue {
        let value = unsafe {
            LLVMConstInt(
                LLVMType::Int1.to_raw_llvm(),
                if val { 1 } else { 0 } as c_ulonglong,
                LLVM_FALSE,
            )
        };
        self.value(value)
    }

    pub fn static_string(&mut self, val: &str) -> LLVMValue {
        unsafe {
            let llvm_str =
                LLVMConstString(self.new_string_ptr(val), val.len() as c_uint, LLVM_FALSE);
            let llvm_array = LLVMAddGlobal(
                self.module,
                LLVMArrayType(LLVMType::Int8.to_raw_llvm(), (val.len() + 1) as c_uint),
                self.empty_name,
            );
            LLVMSetInitializer(llvm_array, llvm_str);
            LLVMSetGlobalConstant(llvm_array, LLVM_TRUE);
            self.value(llvm_array)
        }
    }

    pub fn const_int32(&mut self, val: i32) -> LLVMValue {
        self.value(unsafe {
            LLVMConstInt(
                LLVMType::Int32.to_raw_llvm(),
                val as c_ulonglong,
                LLVM_FALSE,
            )
        })
    }

    // functions

    pub fn declare_function(
        &mut self,
        name: &str,
        ret: LLVMType,
        args: &[LLVMType],
        varargs: bool,
    ) -> LLVMFunction {
        let mut llvm_args = Vec::new();
        for t in args.iter() {
            llvm_args.push(t.to_raw_llvm());
        }
        unsafe {
            let fn_type = LLVMFunctionType(
                ret.to_raw_llvm(),
                llvm_args.as_mut_ptr(),
                llvm_args.len() as u32,
                if varargs { LLVM_TRUE } else { LLVM_FALSE },
            );

            let f = LLVMAddFunction(self.module, self.new_string_ptr(name), fn_type);
            self.function(f)
        }
    }

    pub fn add_block(&mut self, f: LLVMFunction) -> LLVMBasicBlock {
        self.add_block_named(f, LLVMName::Anonymous)
    }

    pub fn add_block_named(&mut self, f: LLVMFunction, name: LLVMName) -> LLVMBasicBlock {
        let f = self.llvm_function(f);
        let block = unsafe { LLVMAppendBasicBlock(f, self.string_ptr_from_name(name)) };
        self.block(block)
    }

    pub fn param(&mut self, f: LLVMFunction, n: usize) -> LLVMValue {
        let f = self.llvm_function(f);
        let n_params = unsafe { LLVMCountParams(f) as usize };
        if n >= n_params {
            panic!("No parameter at index {}", n);
        }
        let param = unsafe { LLVMGetParam(f, n as u32) };
        self.value(param)
    }

    // instructions

    pub fn call(&mut self, bb: LLVMBasicBlock, f: LLVMFunction, args: &[LLVMValue]) -> LLVMValue {
        self.call_named(bb, f, args, LLVMName::Anonymous)
    }

    pub fn call_named(
        &mut self,
        bb: LLVMBasicBlock,
        f: LLVMFunction,
        args: &[LLVMValue],
        name: LLVMName,
    ) -> LLVMValue {
        let f = self.llvm_function(f);
        let mut llvm_args = Vec::new();
        for arg in args.iter() {
            llvm_args.push(self.llvm_value(*arg))
        }
        let builder = self.builder(bb);
        let llvm_value = unsafe {
            LLVMBuildCall(
                builder.builder,
                f,
                llvm_args.as_mut_ptr(),
                llvm_args.len() as c_uint,
                self.string_ptr_from_name(name),
            )
        };
        self.value(llvm_value)
    }

    pub fn br(&mut self, bb: LLVMBasicBlock, next: LLVMBasicBlock) {
        let builder = self.builder(bb);
        unsafe {
            LLVMBuildBr(builder.builder, self.llvm_block(next));
        }
    }

    pub fn conditional_br(
        &mut self,
        bb: LLVMBasicBlock,
        condition: LLVMValue,
        then: LLVMBasicBlock,
        otherwise: LLVMBasicBlock,
    ) {
        let builder = self.builder(bb);
        unsafe {
            LLVMBuildCondBr(
                builder.builder,
                self.llvm_value(condition),
                self.llvm_block(then),
                self.llvm_block(otherwise),
            );
        }
    }

    pub fn pointer_cast(
        &mut self,
        bb: LLVMBasicBlock,
        pointer: LLVMValue,
        ty: LLVMType,
    ) -> LLVMValue {
        self.pointer_cast_named(bb, pointer, ty, LLVMName::Anonymous)
    }

    pub fn pointer_cast_named(
        &mut self,
        bb: LLVMBasicBlock,
        pointer: LLVMValue,
        ty: LLVMType,
        name: LLVMName,
    ) -> LLVMValue {
        let builder = self.builder(bb);
        let pointer = self.llvm_value(pointer);
        let name = self.string_ptr_from_name(name);
        let llvm_value =
            unsafe { LLVMBuildPointerCast(builder.builder, pointer, ty.to_raw_llvm(), name) };
        self.value(llvm_value)
    }

    pub fn ret(&mut self, bb: LLVMBasicBlock, value: LLVMValue) {
        let builder = self.builder(bb);
        let llvm_value = self.llvm_value(value);
        unsafe {
            LLVMBuildRet(builder.builder, llvm_value);
        }
    }

    pub fn ibinop(
        &mut self,
        bb: LLVMBasicBlock,
        binop: LLVMIBinop,
        value1: LLVMValue,
        value2: LLVMValue,
    ) -> LLVMValue {
        self.ibinop_named(bb, binop, value1, value2, LLVMName::Anonymous)
    }

    pub fn ibinop_named(
        &mut self,
        bb: LLVMBasicBlock,
        binop: LLVMIBinop,
        value1: LLVMValue,
        value2: LLVMValue,
        name: LLVMName,
    ) -> LLVMValue {
        let builder = self.builder(bb);
        let value1 = self.llvm_value(value1);
        let value2 = self.llvm_value(value2);
        let name = self.string_ptr_from_name(name);
        let llvm_value = unsafe {
            use llvm_sys::LLVMIntPredicate::*;
            use LLVMIBinop::*;
            match binop {
                // arith
                Add => LLVMBuildAdd(builder.builder, value1, value2, name),
                Sub => LLVMBuildSub(builder.builder, value1, value2, name),
                Mul => LLVMBuildMul(builder.builder, value1, value2, name),
                SDiv => LLVMBuildSDiv(builder.builder, value1, value2, name),
                // logic
                And => LLVMBuildAnd(builder.builder, value1, value2, name),
                Or => LLVMBuildOr(builder.builder, value1, value2, name),
                // cmp
                Eq => LLVMBuildICmp(builder.builder, LLVMIntEQ, value1, value2, name),
                Neq => LLVMBuildICmp(builder.builder, LLVMIntNE, value1, value2, name),
                SLt => LLVMBuildICmp(builder.builder, LLVMIntSLT, value1, value2, name),
                SLe => LLVMBuildICmp(builder.builder, LLVMIntSLE, value1, value2, name),
                SGt => LLVMBuildICmp(builder.builder, LLVMIntSGT, value1, value2, name),
                SGe => LLVMBuildICmp(builder.builder, LLVMIntSGE, value1, value2, name),
            }
        };
        self.value(llvm_value)
    }

    pub fn not(&mut self, bb: LLVMBasicBlock, value: LLVMValue) -> LLVMValue {
        self.not_named(bb, value, LLVMName::Anonymous)
    }

    pub fn not_named(&mut self, bb: LLVMBasicBlock, value: LLVMValue, name: LLVMName) -> LLVMValue {
        let builder = self.builder(bb);
        let value = self.llvm_value(value);
        let name = self.string_ptr_from_name(name);
        let llvm_value = unsafe { LLVMBuildNot(builder.builder, value, name) };
        self.value(llvm_value)
    }

    pub fn alloca(&mut self, bb: LLVMBasicBlock, ty: LLVMType) -> LLVMValue {
        self.alloca_named(bb, ty, LLVMName::Anonymous)
    }

    pub fn alloca_named(&mut self, bb: LLVMBasicBlock, ty: LLVMType, name: LLVMName) -> LLVMValue {
        let builder = self.builder(bb);
        let ty = ty.to_raw_llvm();
        let name = self.string_ptr_from_name(name);
        let llvm_value = unsafe { LLVMBuildAlloca(builder.builder, ty, name) };
        self.value(llvm_value)
    }

    pub fn load(&mut self, bb: LLVMBasicBlock, ptr: LLVMValue) -> LLVMValue {
        self.load_named(bb, ptr, LLVMName::Anonymous)
    }

    pub fn load_named(&mut self, bb: LLVMBasicBlock, ptr: LLVMValue, name: LLVMName) -> LLVMValue {
        let builder = self.builder(bb);
        let ptr = self.llvm_value(ptr);
        let name = self.string_ptr_from_name(name);
        let llvm_value = unsafe { LLVMBuildLoad(builder.builder, ptr, name) };
        self.value(llvm_value)
    }

    pub fn store(&mut self, bb: LLVMBasicBlock, value: LLVMValue, ptr: LLVMValue) {
        let builder = self.builder(bb);
        let ptr = self.llvm_value(ptr);
        let value = self.llvm_value(value);
        unsafe { LLVMBuildStore(builder.builder, value, ptr) };
    }

    // private

    fn builder(&mut self, bb: LLVMBasicBlock) -> Builder {
        Builder::for_block(self.llvm_block(bb))
    }

    fn new_string_ptr(&mut self, s: &str) -> *const i8 {
        if !s.is_ascii() {
            panic!("Can't send non-ascii string {} to LLVM", s);
        }
        let cstring = CString::new(s).unwrap();
        let ptr = cstring.as_ptr() as *const _;
        self.strings.push(cstring);
        ptr
    }

    fn string_ptr_from_name(&mut self, name: LLVMName) -> *const i8 {
        match name {
            LLVMName::Anonymous => self.empty_name,
            LLVMName::Name(s) => self.new_string_ptr(s),
        }
    }

    fn value(&mut self, v: LLVMValueRef) -> LLVMValue {
        self.values.push(v);
        let index = self.values.len() - 1;
        LLVMValue { id: index }
    }

    fn llvm_value(&self, v: LLVMValue) -> LLVMValueRef {
        self.values[v.id]
    }

    fn function(&mut self, f: LLVMValueRef) -> LLVMFunction {
        self.functions.push((f, false));
        let index = self.functions.len() - 1;
        LLVMFunction { id: index }
    }

    fn llvm_function(&self, f: LLVMFunction) -> LLVMValueRef {
        self.functions[f.id].0
    }

    fn block(&mut self, b: LLVMBasicBlockRef) -> LLVMBasicBlock {
        self.blocks.push(b);
        let index = self.blocks.len() - 1;
        LLVMBasicBlock { id: index }
    }

    fn llvm_block(&self, f: LLVMBasicBlock) -> LLVMBasicBlockRef {
        self.blocks[f.id]
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        // Rust requires that drop() is a safe function.
        unsafe {
            LLVMDisposeModule(self.module);
        }
    }
}

pub enum LLVMName<'a> {
    Anonymous,
    Name(&'a str),
}

#[derive(Clone, Debug)]
pub enum LLVMType {
    Int1,
    Int8,
    Int32,
    Pointer(Rc<LLVMType>),
}

#[derive(Clone, Copy, Debug)]
pub struct LLVMBasicBlock {
    id: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct LLVMValue {
    id: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct LLVMFunction {
    id: usize,
}

impl LLVMType {
    pub fn pointer(t: LLVMType) -> LLVMType {
        LLVMType::Pointer(Rc::new(t))
    }

    fn to_raw_llvm(&self) -> LLVMTypeRef {
        use LLVMType::*;
        CONTEXT.with(|c| match self {
            Int1 => unsafe { LLVMInt1TypeInContext(c.context) },
            Int8 => unsafe { LLVMInt8TypeInContext(c.context) },
            Int32 => unsafe { LLVMInt32TypeInContext(c.context) },
            Pointer(pt) => unsafe {
                let inner = pt.to_raw_llvm();
                LLVMPointerType(inner, 0)
            },
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub enum LLVMIBinop {
    Add,
    Sub,
    Mul,
    SDiv,
    And,
    Or,
    Eq,
    Neq,
    SLt,
    SLe,
    SGt,
    SGe,
}

#[cfg(test)]
mod tests {
    use crate::*;
    use std::process::{Command, Output};
    use tempfile::NamedTempFile;

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }

    fn execute_module(module: &mut Module) -> Output {
        let out_file = NamedTempFile::new().expect("temp file creation failed");
        let out_file_path = out_file.path().to_str().unwrap();
        module.write(out_file_path);
        let out = Command::new("lli")
            .arg(out_file_path)
            .output()
            .expect("process failed to execute");
        out
    }

    #[test]
    fn test_basic() {
        let mut module = Module::new("basic");
        let f = module.declare_function("main", LLVMType::Int32, &[], false);
        let four = module.const_int32(4);
        let bb = module.add_block(f);
        module.ret(bb, four);
        let out = execute_module(&mut module);
        assert_eq!(out.status.code(), Some(4));
    }

    #[test]
    #[should_panic]
    fn test_cross_module_panics() {
        let mut module1 = Module::new("mod");
        let f = module1.declare_function("f", LLVMType::Int32, &[], false);
        let bb2 = module1.add_block(f);

        let mut module2 = Module::new("mod2");
        let f = module1.declare_function("f", LLVMType::Int32, &[], false);
        let bb1 = module1.add_block(f);
        module2.br(bb1, bb2);
    }

}
