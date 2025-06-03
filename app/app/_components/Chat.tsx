import { StyleSheet, Text, View } from "react-native";
import React, { memo, PropsWithChildren } from "react";

interface chatInterface {
    role: 'user' | 'bot'
}

function Chat({ children, role }: PropsWithChildren & chatInterface) {
    return (
        <View style={{ ...styles.viewContainer, borderLeftColor: role === 'user' ? '#7dd3fc' : '#fda4af' }}>
            {children}
        </View>
    );
}

const styles = StyleSheet.create({
    viewContainer: {
        padding: 10,
        margin: 10,
        borderLeftWidth: 5,
        backgroundColor: '#e2e8f0',
    }
});


export default memo(Chat)